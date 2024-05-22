import argparse
import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.modules.module import Module


import numpy as np

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if logger.hasHandlers(): 
        logger.handlers.clear() 
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

class CustomDataset(Dataset):
    def __init__(self, input_, output_):
        self.input_ = torch.from_numpy(input_).float()
        self.output_ = torch.from_numpy(output_).float()

    def __len__(self):
        return len(self.input_)

    def __getitem__(self, idx):
        x = self.input_[idx]
        y = self.output_[idx]
        
        return x, y


class DenseRagged(Module):
    def __init__(self, in_dim, out_dim, use_bias=True, activation='linear', **kwargs):
        super(DenseRagged, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation

        if use_bias == True:
            self.linear = nn.Linear(self.in_dim, self.out_dim, bias=True)
        else : 
            self.linear = nn.Linear(self.in_dim, self.out_dim, bias=False)
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, x):
        output = self.linear(x)
        if self.activation is not None:
            output = self.activation(output)
        return output
class DenseNet(Module):
    def __init__(self, units_list, operator='mean', pretrain=False, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        
        self.operator = operator
        self.pretrain = pretrain
        
        self.layer1 = DenseRagged(units_list[0], units_list[1], use_bias=True, activation='relu')
        self.layer2 = DenseRagged(units_list[1], units_list[2], use_bias=True, activation='relu')
        self.layer3 = DenseRagged(units_list[2], units_list[3], use_bias=True, activation='relu')
        self.layer4 = DenseRagged(units_list[3], units_list[4], use_bias=False, activation='relu')
        self.layer5 = DenseRagged(units_list[4], units_list[5], use_bias=False, activation='relu')
        self.layer6 = DenseRagged(units_list[5], units_list[6], use_bias=False, activation='relu')
        self.layer7 = DenseRagged(units_list[6], units_list[7], use_bias=False, activation='sigmoid')
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.pretrain == True:
            if self.operator == 'sum':
                x = torch.sum(x, dim=1)
            elif self.operator == 'mean':
                x = torch.mean(x, dim=1)
        else :
            if self.operator == 'sum':
                x = torch.sum(x, dim=0, keepdim=True)
            elif self.operator == 'mean':
                x = torch.mean(x, dim=0, keepdim=True)

        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        out = self.layer7(x)
        
        return out
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def parse_option():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=25000)
    parser.add_argument('--lr', default=5e-4)
    parser.add_argument('--print_freq', default=200)
    # parser.add_argument('--save_freq', default=1)
    # parser.add_argument('--eval_freq', default=200)
    parser.add_argument('--min_delta', default=1e-10)
    parser.add_argument('--train_ratio', default=0.75)
    parser.add_argument('--units_list', default=[64, 64, 20, 10, 50, 100, 200, 400], nargs='+', type=int)
    parser.add_argument('--operator', default='mean')
    parser.add_argument('--dim', default=0, type=int)
    parser.add_argument('--input_dir', default='ripsnetdata/Teacher56_batch128_FCinput_PCD.npy')
    parser.add_argument('--output_dir', default='ripsnetdata/Teacher56_batch128_FCinput_PI_dim0.npy')
    parser.add_argument('--rips_trial', default=0, type=str)
    parser.add_argument('--norm', default=True, type=str)
    parser.add_argument('--train_bs', default=64, type=int)
    parser.add_argument('--val_bs', default=32, type=int)

    opt = parser.parse_args()
    return opt

units_list_dict = {
    16:[16, 16, 16, 16, 50, 100, 200, 400],
    32:[32, 32, 16, 16, 50, 100, 200, 400],
    64:[64, 64, 32, 32, 50, 100, 200, 400],
    128:[128, 128, 64, 64, 50, 100, 200, 400],
    256:[256, 256, 128, 128, 50, 100, 200, 400],
    512:[512, 512, 256, 256, 50, 100, 200, 400],
    1024:[1024, 512, 256, 256, 50, 100, 200, 400],
    2048:[2048, 1024, 512, 256, 50, 100, 200, 400]
}
    
def main(opt):

    opt.log_dir = 'ripsnet/result/'+opt.input_dir.split('/')[1][:-4]+'_'+opt.output_dir.split('_')[-1][:-4]
    os.makedirs(opt.log_dir, exist_ok=True)

    logger = get_logger(logpath=os.path.join(opt.log_dir, (opt.input_dir.split('/')[-1][:-8]+'_dim_%s'%opt.dim + '.log')), filepath=os.path.abspath(__file__))
    logger.info(opt)
    
    input_ = np.load(opt.input_dir)
    output_ = np.load(opt.output_dir)
    
    opt.units_list = units_list_dict[input_.shape[-1]]

    len_ = int(len(input_)*opt.train_ratio)
    
    train_input = input_[:len_]
    train_output = output_[:len_]
    
    val_input = input_[len_:]
    val_output = output_[len_:]
    
    train_dataset = CustomDataset(train_input, train_output)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.train_bs, shuffle=True)
    
    val_dataset = CustomDataset(val_input, val_output)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.val_bs, shuffle=False)

    model = DenseNet(opt.units_list, operator=opt.operator, pretrain=True)    
    optimizer = optim.Adamax(model.parameters(), opt.lr)
    loss = nn.MSELoss()
    
    if torch.cuda.is_available():
        print('='*10, 'CUDA is available', '='*10)
        model = model.cuda()
        loss = loss.cuda()
    
    logger.info(model)
    logger.info("The number of RipsNet parameters: {}".format(count_parameters(model)))
    
    val_cost_ = np.inf
    val_cost = 0
    counter = 0
    patience =50
    
    print('Begin training !!!!!')
    for epoch in range(opt.epochs):
        model.train()
        for batch_idx, samples in enumerate(train_dataloader):
            x, y = samples

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            
            pred = model(x)
            
            cost = loss(pred, y)
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if batch_idx % opt.print_freq == 0:
                logger.info('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
                        epoch, opt.epochs, batch_idx+1, len(train_dataloader),
                        cost.item()))
        model.eval()
        with torch.no_grad():
            for idx, val_samples in enumerate(val_dataloader):
                val_x, val_y = val_samples
                if torch.cuda.is_available():
                    val_x = val_x.cuda()
                    val_y = val_y.cuda()
                
                output = model(val_x)                
                val_cost += loss(output, val_y)
                
            val_cost = val_cost / len(val_dataloader)
            logger.info('Validation loss: {:.6f}'.format(val_cost.item()))
            
            if val_cost > val_cost_ - opt.min_delta:
                counter += 1
                logger.info(f'EarlyStopping counter: {counter} out of {patience}')
                if counter >= patience:
                    break
            else:
                state = {'epoch' : epoch,
                'model' : model.state_dict(),
                'best_acc' : val_cost,
                'optimizer' : optimizer.state_dict()}
                save_file = os.path.join(opt.log_dir, '%s'%opt.operator+'_best.pth')
                logger.info('saving the best model!')
                torch.save(state, save_file)
                val_cost_ = val_cost
                counter = 0

                    

            val_cost = 0