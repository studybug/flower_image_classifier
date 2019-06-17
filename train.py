import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import pickle

import argparse
#import importlib pm = importlib.__import__('utils')

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('data_dir', action='store', help='train on the data directory')
parser.add_argument('--save_dir', help='the path where to save the checkpoint')
parser.add_argument('--arch', default='vgg16', help='choose a pytorch model')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--hidden_units',  dest="learning_rate", action='store', type=int, default=500)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--gpu', default= "gpu", action='store_true')

args = parser.parse_args()
data_dir = args.data_dir

trainloader, vaildloader, testloader = pickle.load(open(data_dir))

path = pa.save
lr = pa.learning_rate
structure = pa.vgg16
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
model, optimizer, criterion = futils.nn_setup(structure, dropout, hidden_layer1, lr, power)
power = pa.gpu
epochs = pa.epochs
train_network(model, optimizer, criterion, epochs, 20, trainloader, power)
futils,save_checkpoint(path, structure, hidden_layer1, dropout, lr)
print("The model is trained: ready for the gold metal")