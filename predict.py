import numpy as np
import matplotlib.pyplot as plt
import os, random
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import time
import json
import PIL
from PIL import Image
import pickle

import argparse

parser = argparse.ArgumentParser(description='Predict')
parser.add_argument('data_dir', nargs='*', action='store', default="./flowers/")
parser.add_argument('--save_dir', action="store", default=".",dest='save_dir', type=str, help='Directory to save training checkpoint file',)
parser.add_argument('--save_name', action="store", default="checkpoint", dest='save_name', type=str, help='Checkpoint filename.',)
parser.add_argument('--categories_json', action="store", default="cat_to_name.json", dest='categories_json', type=str, help='Path to file containing the categories.')
parser.add_argument('--vgg16', action="store", default="vgg16",dest='arch', type=str, help='Supported architectures')
parser.add_argument('--gpu', action="store_true", dest="use_gpu", default=False, help='Use GPU')
parser.add_argument('--hidden_units', dest="hidden_units", type= int, action='store', default=500)
parser.add_argument('--learning_rate', dest="learning_rate", action='store', default=0.01)
parser.add_argument('--epochs', dest="epochs", action='store', default=1)


training_loader, testing_loader, validation_loader = pickle.load(open(data_dir))

load_checkpoint(path)


with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)


probabilities = predict(path_image, model, number_of_outputs, power)


labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])


i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1

print("Results")