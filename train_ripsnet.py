# Train RipsNet
import argparse
from ast import parse
from re import L
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.modules.module import Module

import numpy as np

from RipsNet_modules import parse_option, main

model_name = 'resnet56'
batch_size = 64
location = 'FCinput'
dim = 'dim0.npy'

opt = parse_option()
opt.input_dir = 'ripsnet/ripsnetdata/cifar100_{}_batch{}_{}_PCD.npy'.format(model_name, batch_size, location)
opt.output_dir = 'ripsnet/ripsnetdata/cifar100_{}_batch{}_{}_PI_'.format(model_name, batch_size, location) + dim
opt.operator = 'mean'

main(opt)