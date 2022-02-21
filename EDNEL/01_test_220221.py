"""
22.02.21 EDNEL
Custom Device import test
"""

import os
from time import time

# Import from Pytorch
import torch
from torch import nn
from torchvision import datasets, transforms

# Import from aihwkit
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import UnitCellRPUConfig
from aihwkit.simulator.configs.devices import (TransferCompound, SoftBoundsDevice)

from aihwkit.simulator.rpu_base import cuda

# Import matplotlib
import matplotlib.pyplot as plt

# Check device
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1

DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# Path where the datasets will be stored
PATH_DATASET = os.path.join('data,', 'DATASET')

#Network definition
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10 

#Training hyperparameters
EPOCHS = 30
BATCH_SIZE = 64



def load_images():
    """ Load images for train from the torchvision datasets"""
    transform = transforms.Compose([transforms.ToTensor()])

    #Load the images
    train_set = datasets.MNIST(PATH_DATASET, download=True, train =True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train =False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle =True)

    