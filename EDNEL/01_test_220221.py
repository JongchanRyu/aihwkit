"""
22.02.21 EDNEL
Custom Device import test
"""

from inspect import Traceback
import os
from datetime import datetime
from numpy import datetime_as_string

# Import from Pytorch
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

# Import from aihwkit
from aihwkit.nn import AnalogLinear
from aihwkit.nn.modules.container import AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import UnitCellRPUConfig
from aihwkit.simulator.configs.devices import (TransferCompound, SoftBoundsDevice)

# Import Custom device setting
from aihwkit.simulator.configs.devices import CustomDevice
from aihwkit.simulator.presets.configs import GokmenVlasovPreset

# Import CUDA
from aihwkit.simulator.rpu_base import cuda

# Import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Check device
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1

DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# Path where the datasets will be stored
PATH_DATASET = os.path.join('data', 'DATASET')

# Path where the results will be stored
PATH_RESULT = os.path.join('results', 
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
DATETIME = datetime.now()

#Network definition
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10 

#Training hyperparameters
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.05
SEED = 1

#Define Tiki-taka based rpu configuration
# RPU_CONFIG = UnitCellRPUConfig(
#     device = TransferCompound(
        
#         # Device consist of the Tiki-taka compound
#         #unit_cell_devices=[
#             #CustomDevice()
#         # Devices that compose the Tiki-taka compound.
#         unit_cell_devices=[
#             SoftBoundsDevice(w_min=-0.3, w_max=0.3),
#             SoftBoundsDevice(w_min=-0.6, w_max=0.6)
#         ],

#         # Make some adjustments of the way Tiki-Taka is performed.
#         units_in_mbatch=True,    # batch_size=1 anyway
#         transfer_every=2,        # every 2 batches do a transfer-read
#         n_reads_per_transfer=1,  # one forward read for each transfer
#         gamma=0.0,               # all SGD weight in second device
#         scale_transfer_lr=True,  # in relative terms to SGD LR
#         transfer_lr=1.0,         # same transfer LR as for SGD
#         fast_lr=0.1,             # SGD update onto first matrix constant
#         transfer_columns=True    # transfer use columns (not rows)

#     )
# )

RPU_CONFIG = GokmenVlasovPreset()

# Load images from torchvision package
def load_images():
    """ Load images for train from the torchvision datasets"""
    transform = transforms.Compose([transforms.ToTensor()])

    #Load the images
    train_set = datasets.MNIST(PATH_DATASET, download=True, train =True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train =False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle =True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle =True)

    return train_data, validation_data

def create_DNN_analog_network(rpu_config = RPU_CONFIG):
    """ Return DNN for MNIST consist of analog model with RPU_CONFIG
    
    Args:
        RPU_CONFIG : rpu configuration of each analog layer 
    
    Returns:
        nn.module: created analog model
    """
    model = AnalogSequential(
        AnalogLinear(INPUT_SIZE, HIDDEN_SIZES[0], bias=True, rpu_config=RPU_CONFIG),
        nn.Sigmoid(),
        AnalogLinear(HIDDEN_SIZES[0], HIDDEN_SIZES[1], bias=True, rpu_config=RPU_CONFIG),
        nn.Sigmoid(),
        AnalogLinear(HIDDEN_SIZES[1], OUTPUT_SIZE, bias=True, rpu_config=RPU_CONFIG),
        nn.LogSoftmax(dim=1)
    )

    if USE_CUDA:
        model.cuda()
    
    print(model)

    return model

def create_sgd_optimizer(model, learning_rate):
    """
    Create the analog optimizer
    """
    optimizer = AnalogSGD(model.parameters(), lr = learning_rate)
    optimizer.regroup_param_groups(model)

    return optimizer

def train_step(train_data, model, criterion, optimizer):
    """Train network.

    Args:
        train_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer

    Returns:
        nn.Module, nn.Module, float:  model, optimizer and loss for per epoch
    """
    total_loss = 0

    model.train()

    for images, labels in train_data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        #Flatten process
        images = images.view(images.shape[0], -1)

        optimizer.zero_grad()

        # Add training Tensor to the model (input).
        output = model(images)
        loss = criterion(output, labels)

        # Run training (backward propagation).
        loss.backward()

        # Optimize weights.
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    epoch_loss = total_loss / len(train_data.dataset)

    return model, optimizer, epoch_loss

def test_evaluation(validation_data, model, criterion):
    """Test trained network.

    Args:
        validation_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss

    Returns:
        nn.Module, float, float, float:  model, loss, error, and accuracy
    """
    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()

    for images, labels in validation_data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        #Flatten process
        images = images.view(images.shape[0], -1)

        pred = model(images)
        loss = criterion(pred, labels)
        total_loss += loss.item() * images.size(0)

        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()
        accuracy = predicted_ok/total_images*100
        error = (1-predicted_ok/total_images)*100

    epoch_loss = total_loss / len(validation_data.dataset)

    return model, epoch_loss, error, accuracy

def training_loop(model, criterion, optimizer, train_data, validation_data, epochs, print_every=1):
    """Training loop.

    Args:
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer
        train_data (DataLoader): Validation set to perform the evaluation
        validation_data (DataLoader): Validation set to perform the evaluation
        epochs (int): global parameter to define epochs number
        print_every (int): defines how many times to print training progress

    Returns:
        nn.Module, Optimizer, Tuple: model, optimizer,
            and a tuple of train losses, validation losses, and test
            error
    """
    train_losses = []
    valid_losses = []
    test_error = []

    # Train model
    for epoch in range(0, epochs):
        # Train_step
        model, optimizer, train_loss = train_step(train_data, model, criterion, optimizer)
        train_losses.append(train_loss)

        # Validate_step
        with torch.no_grad():
            model, valid_loss, error, accuracy = test_evaluation(
                validation_data, model, criterion)
            valid_losses.append(valid_loss)
            test_error.append(error)

        if epoch % print_every == (print_every - 1):
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Test error: {error:.2f}%\t'
                  f'Accuracy: {accuracy:.2f}%\t')

    # Save results
    np.savetxt(os.path.join(PATH_RESULT, "Test_error.csv"), test_error, delimiter=",")
    np.savetxt(os.path.join(PATH_RESULT, "Train_Losses.csv"), train_losses, delimiter=",")
    np.savetxt(os.path.join(PATH_RESULT, "Valid_Losses.csv"), valid_losses, delimiter=",")
    
    return (model, optimizer), train_losses, valid_losses, test_error

def plot_results(train_losses, valid_losses, test_error):
    """Plot results.

    Args:
        train_losses (List): training losses as calculated in the training_loop
        valid_losses (List): validation losses as calculated in the training_loop
        test_error (List): test error as calculated in the training_loop
    """
    fig = plt.plot(train_losses, 'r-s', valid_losses, 'b-o')
    plt.title('aihwkit MNIST')
    plt.legend(fig[:2], ['Training Losses', 'Validation Losses'])
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.grid(which='both', linestyle='--')
    plt.savefig(os.path.join(PATH_RESULT, 'test_losses.png'))
    plt.close()

    fig = plt.plot(test_error, 'r-s')
    plt.title('aihwkit MNIST')
    plt.legend(fig[:1], ['Validation Error'])
    plt.xlabel('Epoch number')
    plt.ylabel('Test Error [%]')
    plt.yscale('log')
    plt.ylim((5e-1, 1e2))
    plt.grid(which='both', linestyle='--')
    plt.savefig(os.path.join(PATH_RESULT, 'test_error.png'))
    plt.close()


def main():
    os.makedirs(PATH_RESULT, exist_ok=True)
    torch.manual_seed(SEED)

    #Load datasets.
    train_data, validation_data = load_images()

    # Prepare the model.
    model = create_DNN_analog_network()

    print(model)

    optimizer = create_sgd_optimizer(model, LEARNING_RATE)

    criterion = nn.CrossEntropyLoss()

    _, train_losses, valid_losses, test_error = training_loop(model, criterion, optimizer, train_data, 
                                                                validation_data, EPOCHS, print_every=1)
    
    #Plot and save results
    plot_results(train_losses, valid_losses, test_error)

    print(f'{datetime.now().time().replace(microsecond=0)} ---'
          f'Completed')


if __name__ == '__main__':
    # Excute only if run as the entry point into the program
    main()
