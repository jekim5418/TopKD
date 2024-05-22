from __future__ import print_function

# from topologylayer.nn import AlphaLayer, RipsLayer

import torch
import torch.nn as nn

import numpy as np

class DenseRagged(nn.Module):
    def __init__(self, in_dim, out_dim, use_bias=True, activation='linear', **kwargs):
        super(DenseRagged, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        if use_bias == True:
            self.linear = nn.Linear(self.in_dim, self.out_dim, bias=True)
        else : 
            self.linear = nn.Linear(self.in_dim, self.out_dim, bias=False)

    def forward(self, x):
        output = self.linear(x)
        if self.activation is not None:
            output = self.activation(output)
        return output
    
class DenseNet(nn.Module):
    def __init__(self, input_shape, output_shape, units_list, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        self.layer1 = DenseRagged(input_shape, units_list[0], use_bias=True, activation='relu')
        self.layer2 = DenseRagged(units_list[0], units_list[1], use_bias=True, activation='relu')
        self.layer3 = DenseRagged(units_list[1], units_list[2], use_bias=True, activation='relu')
        self.layer4 = DenseRagged(units_list[2], units_list[3], use_bias=False, activation='relu')
        self.layer5 = DenseRagged(units_list[3], units_list[4], use_bias=False, activation='relu')
        self.layer6 = DenseRagged(units_list[4], units_list[5], use_bias=False, activation='relu')
        self.layer7 = DenseRagged(units_list[5], output_shape, use_bias=False, activation='sigmoid')
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.mean(x, dim=1)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        out = self.layer7(x)
        
        return out


class TDALoss(nn.Module):
    """Distilling the TDA of feature in a Neural Network"""
    def __init__(self, s_shape, t_shape, output_shape, opt):
        super(TDALoss, self).__init__()
        self.opt = opt
        self.densenet = DenseNet(s_shape, output_shape,)
        self.criterion  = nn.MSELoss()
        self.embedding = PointPooling()

    def forward(self, feat_t, feat_s):
        dgm_t, _ = self.method(feat_t)
        dgm_s, _ = self.method(feat_s)
        
        emb_t = self.embedding(dgm_t)
        emb_s = self.embedding(dgm_s)
        
        return self.criterion(emb_s, emb_t)


class PointPooling(nn.Module):
    def __init__(self, dim_input=2, dim_hidden=32, dim_output=32):
        super(PointPooling, self).__init__()
        self.weight = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(dim_hidden, 1),
        )
        
        self.embedding = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(dim_hidden, dim_output),
        )
        
    def forward(self, diagrams):
        ret = []
        for dgm in diagrams:
            dgm = torch.nan_to_num(dgm, posinf=0)
            weight = self.weight(dgm)
            embedding = self.embedding(dgm)
            pooling = torch.mean(weight * embedding, dim=0)
            ret.append(pooling)
        
        return torch.cat(ret)