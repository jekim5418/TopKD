import os
import torch
import torch.nn as nn
from model.resnet import resnet56, resnet32x4, resnet110
from model.vgg import VGG13_bn
from model.resnetv2 import ResNet50
from model.wrn import wrn_40_2
import numpy as np
import gudhi as gd
import pickle
from ripsnet_utils import dataset_to_dg, dg_to_PI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataset.cifar100 import get_cifar100_dataloaders
batch_size = 64
dataloader, _, num_train = get_cifar100_dataloaders(batch_size=batch_size, drop_last=True, is_instance=True)

# Load Teacher model
model_name = 'resnet56'
dir_ckpt = 'save_t_models/resnet56_cifar100_epoch_240_bs_64_lr_0.05_decay_0.0005_trial_5/{}_best.pth'.format(model_name)
state_dict = torch.load(dir_ckpt, map_location=torch.device('cuda'))
model = resnet56(num_classes=100)
model.load_state_dict(state_dict['model'])
model.to(device)

# Produce PCDs
nb_epochs = 256
num = num_train//batch_size
for i,(x,y,z) in enumerate(dataloader):
        emb_fea, logits = model(x.to(device),is_feat=True)
        break
dim_latent_space = emb_fea[-1].size(1)
dataset_FCinput=np.zeros((num*nb_epochs, batch_size, dim_latent_space))
with torch.no_grad():
    for epoch in range(nb_epochs):
        for i,(x,y,z) in enumerate(dataloader):
            emb_fea, logits = model(x.to(device),is_feat=True)
            dataset_FCinput[i+epoch*num] = emb_fea[-1].cpu().numpy().reshape(batch_size, dim_latent_space)

if not os.path.exists('ripsnet/ripsnetdata'):
    os.makedirs('ripsnet/ripsnetdata')
filename = 'ripsnet/ripsnetdata/cifar100_{}_batch{}_{}_PCD.npy'.format(model_name, batch_size, 'FCinput')
np.save(filename, dataset_FCinput)

# Produce PIs
filename = 'ripsnet/ripsnetdata/cifar100_{}_batch{}_{}_PCD.npy'.format(model_name, batch_size, 'FCinput')
dataset_FCinput = np.load(filename)
PD_train0 = dataset_to_dg(dataset_FCinput, h_dim=0)
PI_train0 = dg_to_PI(PD_train0)
# if use dim1
# PD_train0, PD_train1 = dataset_to_dg(dataset_FCinput, h_dim=1)
# PI_train0, PI_train1 = dg_to_PI(PD_train0), dg_to_PI(PD_train1)
filename = 'ripsnet/ripsnetdata/cifar100_{}_batch{}_{}_PI_dim{}.npy'.format(model_name, batch_size, 'FCinput', 0)
np.save(filename, PI_train0)
# filename = '/ripsnet/ripsnetdata/cifar100_{}_batch{}_{}_PI_dim{}.npy'.format(model_name, batch_size, 'FCinput', 1)
# np.save(filename, PI_train1)