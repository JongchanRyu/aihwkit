"""
EDNEL
File created: 22.02.23
CustomDevice test for MNIST dataset

Todo:
    torch.manual_seed(SEED) (if needed)
    import dataset by load_images()
    Prepare the model by create_DNN_analog_network()
    Prepare the optimizer by create_sgd_optimizer(model, learning_rate)
    Set the loss function (default: nn.crossentropyloss)
    Train by training_loop

    optional:
    plot_results
    save_results

Example code at DNNsetup.main()

"""
from DNNsetup import *

# Import from torch
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

# Import configuration module from aihwkit
from aihwkit.simulator.configs import UnitCellRPUConfig
from aihwkit.simulator.configs.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import (ConstantStepDevice, TransferCompound, SoftBoundsDevice)

# Import Custom device setting
from aihwkit.simulator.configs.devices import CustomDevice


def DNN_train_full(custom_config, return_data = False):
    torch.manual_seed(SEED)
    
    train_data, validation_data = load_images()

    model = create_DNN_analog_network(rpu_config=custom_config)
    optimizer = create_sgd_optimizer(model, LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    _, _, train_losses, valid_losses, test_error = training_loop(model, criterion, optimizer, train_data, 
                                                                validation_data, EPOCHS, print_every=1)
    
    #Plot and save results
    plot_results(train_losses, valid_losses, test_error)
    save_results(model, train_losses, valid_losses, test_error)

    if return_data:
        return train_losses, valid_losses, test_error
    

def main():
    custom_rpu_config = SingleRPUConfig(device = CustomDevice(num_sectors=2, ))



if __name__ == '__main__':
    main()