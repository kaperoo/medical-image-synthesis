"""
This file contains the utility functions needed for GANs.

Date:
    August 15, 2020

Project:
    XAI-GAN

Contact:
    explainable.gan@gmail.com
"""

import numpy as np
from torch import Tensor, from_numpy, randn, full
import torch
import torch.nn as nn
from torch.autograd.variable import Variable


def images_to_vectors(images: Tensor) -> Tensor:
    """ converts (Nx199x546) tensor to (Nx108654) torch tensor """
    return images.view(images.size(0), 199 * 546)

def values_target(size: tuple, value: float, cuda: False) -> Variable:
    """ returns tensor filled with value of given size """
    result = Variable(full(size=size, fill_value=value))
    if cuda:
        result = result.cuda()
    return result

def normalize_vector(vector: torch.tensor) -> torch.tensor:
    """ normalize np array to the range of [0,1] and returns as float32 values """
    vector -= vector.min()
    vector /= vector.max()
    vector[torch.isnan(vector)] = 0
    return vector.type(torch.float32)