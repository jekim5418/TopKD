import os
import argparse
import random
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from dataset.cifar10 import get_cifar10_dataloaders
from dataset.cifar100 import get_cifar100_dataloaders
# from dataset.imagenet import get_imagenet_dataloader
from model import model_dict
from tqdm import tqdm
import math

from util import adjust_learning_rate, accuracy, AverageMeter
from loops import train_vanilla as train, validate

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
    parser.add_argument('--model', type=str, default='resnet8')

    parser.add_argument('--epochs', default=240, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=1, type=int)

    # optimization
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--lr_decay_epochs', default=[150, 180, 210], nargs='+', type=int)
    parser.add_argument('--lr_decay_rate', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)


    parser.add_argument('--trial', default=0, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    opt=parser.parse_args()

    opt.model_path='./save_t_models/'
    opt.model_name='{}_{}_epoch_{}_bs_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.epochs, opt.batch_size, opt.lr,
                                                            opt.weight_decay, opt.trial)
    opt.save_folder=os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    
    opt = parse_option()

    wandb.init(project='scratch_cifar', name=opt.model_name)
    wandb.config.update(opt)

    print(opt)

    best_acc = 0

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader=get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
        img_size = 32
    else:
        raise NotImplementedError(opt.dataset)

    #model
    if opt.model == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(nn.Linear(2048, n_cls))        
    else:
        model = model_dict[opt.model](num_classes = n_cls)
    

    wandb.watch(model)

    print('model summary :')
    print(model)

    print('number of teacher model parameters :', count_parameters(model), '\n')

    module_list = nn.ModuleList([])
    trainable_list = nn.ModuleList([])

    module_list.append(model)
    trainable_list.append(model)

    print('number of total trainable parameters :', count_parameters(trainable_list))

    #optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr = opt.lr,
                          momentum = opt.momentum,
                          weight_decay = opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        module_list = module_list.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True


    # train
    for epoch in range(1, opt.epochs + 1):
        
        adjust_learning_rate(epoch, opt, optimizer)
        print('==> training...')

        start_time = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion, optimizer, opt)

        fin_time = time.time()
        print('epoch : {}, total_time : {:.2f}'.format(epoch, fin_time-start_time))

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        # save the model according to save_freq
        if epoch % opt.save_freq == 0:
            print('==> saving...')
            state = {
                'epoch' : epoch,
                'model' : model.state_dict(),
                'train accuracy' : train_acc,
                'train loss' : train_loss,
                'test accuracy' : test_acc,
                'test loss' : test_loss,
                'optimizer' : optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch = epoch))
            torch.save(state, save_file)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch' : epoch,
                'model' : model.state_dict(),
                'best_acc' : best_acc,
                'optimizer' : optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

    print('best accuracy :', best_acc)

    # save the last model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


if __name__ == '__main__':
    
    random_seed = random.randint(1, 1000)
    # random_seed = int(469)
    print('random seed :', random_seed)

    for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
        set_random_seed(random_seed)

    main()

    
    
