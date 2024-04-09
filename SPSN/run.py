import torch
import random
import numpy as np
from datetime import datetime

import torch.utils
import torch.utils.data
from network import create_network, train, test

import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import autograd
from snntorch import spikegen

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

import dataset

device = torch.device('cuda' if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(device)

params = {
    'neuron': "ParaLIF-SB",
    "dir": "nmnist/",
    "dataset": "nmnist",
    "window_size": 1e-3,
    "batch_size": 256,
    "nb_epochs": 2,
    'recurrent': False,
    'nb_layers': 2,
    'hidden_size': 128,
    'data_augmentation': False,
    'input_size': 28*28,
    'nb_class': 10,
    'loss_mode': 'mean',
    'tau_mem': 2e-2,
    'tau_syn': 2e-2,
    'lr': 0.001,
    'weight_decay': 0.,
    'reg_thr': 0.,
    'reg_thr_r': 0.,
    'shift': 0.05,
    'scale': 0.15,
    'num_steps': 312
}

# MNIST dataset
train_dataset = dataset.MNISTCustomDataset(folder='data/MNIST/raw',
                                           train=True,
                                           num_steps=params['num_steps'])

test_dataset = dataset.MNISTCustomDataset(folder='data/MNIST/raw',
                                          train=False,
                                          num_steps=params['num_steps'])

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=params['batch_size'],
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=params['batch_size'],
                                          shuffle=False)

model = create_network(params, device)

print("\n-- Training --\n")

'''x, y = next(iter(train_loader))
print(torch.unique(x))
print(y.dtype)
print(x.dtype)
#x = x.view(x.shape[0], -1)

x = spikegen.rate(x[0], num_steps=312)
#print(x[:, 0, 0].shape)
print(x.shape)
x = x.view(312, 28, 28)

print(y[0])
fig, ax = plt.subplots()
anim = splt.animator(x, fig, ax)
plt.show()
HTML(anim.to_html5_video())
print(x.shape)'''

train_results = train(model, train_loader, nb_epochs=params['nb_epochs'], loss_mode=params['loss_mode'],
                      reg_thr=params['reg_thr'], reg_thr_r=params['reg_thr_r'], lr=params['lr'], weight_decay=params['weight_decay'])

print("\n-- Testing --\n")
test_results = test(model, test_loader, loss_mode=params['loss_mode'])
