import os
import torch
import random
import copy
import numpy as np
import torch.nn as nn
from Cycle_GAN.Cycle_config import batch_size, num_workers, lr, num_epochs, lambda_identity, load_model, save_model, checkpoint_gen_A, checkpoint_gen_B, data_dir_train, data_dir_test, device


def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If dont do this it will have only old learning_rate (not updated) of the old __Checkpoint__,
    # Will relult in a __Bug__ (Update learning rate of __optimizer__(Adam) to the new learning rate __Manualy__)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
