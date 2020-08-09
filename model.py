import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

""" BN stand for Bayesian Network """
def call_bn(bn, x):
    return bn(x)

class CNN(nn.Module):
    pass