import os
import argparse
import random
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from dataset.cifar100 import get_cifar100_dataloaders
from distillation import HintLoss

from model import model_dict

# import model.backbone as backbone

# import metric.loss as loss

from tqdm import tqdm
from torch.utils.data import DataLoader


from util import adjust_learning_rate, accuracy, AverageMeter
from loops import train_distill as train, validate

from distillation import DistillKL
from layers import DenseNet

import wandb


def parse_option():

    parser=argparse.ArgumentParser()
    LookupChoices=type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

    parser.add_argument('--load', 
                        default=None)
    parser.add_argument('--dataset',
                        choices=dict(cifar10='cifar10',
                                    cifar100='cifar100',
                                    imagenet='imagenet'),
                        default='cifar100',
                        action=LookupChoices)
    parser.add_argument('--model_s', type=str, default='resnet8')
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    parser.add_argument('--distill', choices=['KD', 'hint_L2', 'hint_KD'], default='KD', type=str)
    parser.add_argument('--hint_layer', default=None, type=int, help='layer for intermediate mimic feature')
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    parser.add_argument('--epochs', default=240, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=1, type=int)

    # optimization
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--lr_decay_epochs', default=[150, 180, 210], nargs='+', type=int)  ### 0.625, 0.75, 0.875
    parser.add_argument('--lr_decay_rate', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)


    parser.add_argument('--trial', default=0, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    
    ##### TDA argument ####
    parser.add_argument('--TDA', default=True, type=str)
    parser.add_argument('-d', '--delta', type=float, default=None, help='weight balance for tda losses')
    parser.add_argument('--tda_layer_t', default=None, type=int, help='teacher layer for mimic persistend diagram distance')
    parser.add_argument('--tda_layer_s', default=None, type=int, help='student layer for mimic persistend diagram distance')
    parser.add_argument('--dim', default='0', type=str, help='topology matching dimension')
    parser.add_argument('--operator', default='sum')
    parser.add_argument('--units_list', default=[64, 30, 20, 10, 50, 100, 200, 400], nargs='+', type=int)
    parser.add_argument('--rips_path', default='ripsnet/result')


    opt=parser.parse_args()

    opt.model_t = get_teacher_name(opt.path_t)

    if opt.TDA == 'True':
        if opt.tda_layer_s is None:
            opt.tda_layer_s = opt.tda_layer_t
        opt.model_path='./save_s_tda_ripsnet_t_models_final/'
        opt.model_name='T:{}_S:{}_{}_{}_KD:{}_T:{}_epochs_{}_lr_{}_decay_{}_r:{}_a:{}_b:{}_d:{}_tda_layer_t:{}_tda_layer_s:{}_dim_{}_{}_trial_{}'.format(opt.model_t, opt.model_s, opt.dataset, opt.batch_size, opt.distill, opt.kd_T,
                    opt.epochs, opt.lr, opt.weight_decay, opt.gamma, opt.alpha, opt.beta, opt.delta, opt.tda_layer_t, opt.tda_layer_s, opt.dim, opt.operator, opt.trial)
    else :
        opt.model_path='./save_s_models_final/'
        opt.model_name='T:{}_S:{}_{}_KD:{}_T:{}_epochs_{}_lr_{}_decay_{}_r:{}_a:{}_b:{}_trial_{}'.format(opt.model_t, opt.model_s, opt.dataset, opt.distill, opt.kd_T,
                        opt.epochs, opt.lr, opt.weight_decay, opt.gamma, opt.alpha, opt.beta, opt.trial)

    opt.save_folder=os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def main():
    
    opt = parse_option()
    
    wandb.init(project="save_s_tda_ripsnet_t", name=opt.model_name)
    wandb.config.update(opt)
    
    print(opt)

    best_acc = 0

    # dataloader

    # if opt.dataset == 'cifar10':
    #     train_loader, val_loader, n_data=get_cifar10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=True)
    #     n_cls = 10
    if opt.dataset == 'cifar100':
        train_loader, val_loader, n_data=get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=True)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    #model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes = n_cls)
    
    wandb.watch(model_t)
    wandb.watch(model_s)


    syn_data = torch.rand(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()

    feat_t, _ = model_t(syn_data, is_feat=True)
    feat_s, _ = model_s(syn_data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    if opt.distill == 'KD':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint_L2':
        criterion_kd = nn.MSELoss()
    elif opt.distill == 'hint_KD':
        criterion_kd = DistillKL(opt.kd_T)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)
    criterion_list.append(criterion_div)
    criterion_list.append(criterion_kd)
    
    
    
    ###### Apply TDA loss #######
    if opt.TDA =='True':
        rips_pool_t = None
        rips_pool_s = None
        rips_1dconv = None
            
        if opt.tda_layer_t == 0:
            s = 'output'
        else :
            if model_t.nChannels[opt.tda_layer_t] != model_s.nChannels[opt.tda_layer_s]:
                rips_1dconv = nn.Conv2d(model_s.nChannels[opt.tda_layer_s], model_t.nChannels[opt.tda_layer_t], 1, 1)
                rips_1dconv.cuda()
                trainable_list.append(rips_1dconv)
                
            if opt.tda_layer_t == -1:
                s = 'FCinput' 
            elif (opt.tda_layer_t == -4) | (opt.tda_layer_t == -5) :   
                s = 'layer%sinput' % (opt.tda_layer_t + 6)
                rips_pool_t = nn.AvgPool2d(32).cuda()                
                
                if (opt.model_s == 'resnet110x2') & (opt.tda_layer_t == -5):
                    rips_pool_s = nn.AvgPool2d(32).cuda()   
                elif (opt.model_s == 'resnet110x2') & (opt.tda_layer_t == -4):
                    rips_pool_s = nn.AvgPool2d(16).cuda()   
                
                if ((opt.model_s == 'mobilenetv2_1_0')|(opt.model_s == 'vgg8')) & (opt.tda_layer_t == -5):
                    rips_pool_s = nn.AvgPool2d(16).cuda()   
                elif ((opt.model_s == 'mobilenetv2_1_0')|(opt.model_s == 'vgg8')) & (opt.tda_layer_t == -4):
                    rips_pool_s = nn.AvgPool2d(8).cuda()   
                    
                if ((opt.model_s == 'shufflev2') | (opt.model_s == 'shufflev1')) & (opt.tda_layer_t == -5):
                    rips_pool_s = nn.AvgPool2d(32).cuda()   
                elif ((opt.model_s == 'shufflev2') | (opt.model_s == 'shufflev1')) & (opt.tda_layer_t == -4):
                    rips_pool_s = nn.AvgPool2d(16).cuda() 
                    
                if opt.model_t == 'vgg13':
                    rips_pool_t = rips_pool_s
                    
            elif opt.tda_layer_t == -3 :
                s = 'layer%sinput' % (opt.tda_layer_t + 6)
                rips_pool_t = nn.AvgPool2d(16).cuda()
                
                if ((opt.model_s == 'mobilenetv2_1_0')|(opt.model_s == 'vgg8')):
                    rips_pool_s = nn.AvgPool2d(4).cuda()   
                elif ((opt.model_s == 'shufflev2') | (opt.model_s == 'shufflev1')):
                    rips_pool_s = nn.AvgPool2d(8).cuda() 
                    
                if opt.model_t == 'vgg13':
                    rips_pool_t = rips_pool_s
            
            if ((opt.model_s == 'resnet20')) & ((opt.tda_layer_s == -5)|(opt.tda_layer_s == -4)):
                rips_pool_s = nn.AvgPool2d(32).cuda() 
                if (opt.model_t == 'resnet56')&((opt.tda_layer_t == -5)|(opt.tda_layer_t == -4)):
                    rips_pool_t = rips_pool_s
                elif (opt.model_t == 'resnet56')|((opt.tda_layer_t == -3)):
                    rips_pool_t = nn.AvgPool2d(16).cuda() 
                    
            elif ((opt.model_s == 'resnet20')) & (opt.tda_layer_s == -3):
                rips_pool_s = nn.AvgPool2d(16).cuda()
                if (opt.model_t == 'resnet56')&((opt.tda_layer_t == -5)|(opt.tda_layer_t == -4)):
                    rips_pool_t = nn.AvgPool2d(32).cuda()
                elif (opt.model_t == 'resnet56')|((opt.tda_layer_t == -3)):
                    rips_pool_t = rips_pool_s

                    
            trainable_list.append(rips_pool_t)
            trainable_list.append(rips_pool_s)
        opt.rips_ckpt = os.path.join('%s'%(opt.rips_path), '%s_%s_batch%s_%s_PCD_dim%s/%s_best.pth'%(opt.dataset, str(get_teacher_name(opt.path_t)), opt.batch_size, s, opt.dim, opt.operator))
        
        ripsnet_t = DenseNet(opt.units_list, operator=opt.operator, pretrain=False)
        
        wandb.watch(ripsnet_t)

        ripsnet_t.load_state_dict(torch.load(opt.rips_ckpt)['model'])
        for param in ripsnet_t.parameters():
            param.requires_grad=False
        criterion_tda = nn.MSELoss()
        if torch.cuda.is_available():
            ripsnet_t.cuda()
            criterion_tda.cuda()
            
    #optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr = opt.lr,
                          momentum = opt.momentum,
                          weight_decay = opt.weight_decay)

    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list = module_list.cuda()
        criterion_list = criterion_list.cuda()
        cudnn.benchmark = True
        

    # vlidate teacher
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teahcer accuracy :', teacher_acc)

    # train
    for epoch in range(1, opt.epochs + 1):
        
        adjust_learning_rate(epoch, opt, optimizer)
        print('==> training...')

        start_time = time.time()
        if opt.TDA =='True':
            train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt, ripsnet_t, criterion_tda, rips_pool_t, rips_pool_s, rips_1dconv)    
        else :
            train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)

        fin_time = time.time()
        print('epoch : {}, total_time : {:.2f}'.format(epoch, fin_time-start_time))

        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)
        
        # save the model according to save_freq
        if rips_1dconv is None :
            if epoch % opt.save_freq == 0:
                print('==> saving...')
                state = {
                    'epoch' : epoch,
                    'model_s' : model_s.state_dict(),
                    'accuracy' : test_acc,
                    'optimizer' : optimizer.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch = epoch))
                torch.save(state, save_file)

            # save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch' : epoch,
                    'model' : model_s.state_dict(),
                    'best_acc' : best_acc,
                    'optimizer' : optimizer.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
                print('saving the best model!')
                torch.save(state, save_file)
        else :
            if epoch % opt.save_freq == 0:
                print('==> saving...')
                state = {
                    'epoch' : epoch,
                    'model_s' : model_s.state_dict(),
                    'rips_1dconv' : rips_1dconv.state_dict(),
                    'accuracy' : test_acc,
                    'optimizer' : optimizer.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch = epoch))
                torch.save(state, save_file)

            # save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch' : epoch,
                    'model' : model_s.state_dict(),
                    'rips_1dconv' : rips_1dconv.state_dict(),
                    'best_acc' : best_acc,
                    'optimizer' : optimizer.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
                print('saving the best model!')
                torch.save(state, save_file)

    print('best accuracy :', best_acc)

    # save the last model
    if rips_1dconv is None :
        state = {
            'opt': opt,
            'model': model_s.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
        torch.save(state, save_file)
    else :
        state = {
            'opt': opt,
            'model': model_s.state_dict(),
            'rips_1dconv' : rips_1dconv.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
        torch.save(state, save_file)



if __name__ == '__main__':
    
    random_seed = random.randint(1, 1000)
    print('random seed :', random_seed)

    for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
        set_random_seed(random_seed)

    main()

    
    