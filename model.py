import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

# BN stand for Batch Normalization
def call_bn(bn, x):
    """Caller for callable BarchNorm2d
    """
    return bn(x)

class CNN(nn.Module):
    """Convolutional neural network used in this paper.

    There are 9 layers
    """
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False):
        # The input_channel is default at 3 because RGB color (maybe)
        self.input_channel = input_channel
        self.top_bn = top_bn
        self.dropout_rate = dropout_rate
        super().__init__()
        self.c1 = nn.Conv2d(input_channel, 128, 3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.c8 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.c9 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.l_c1 = nn.Linear(128, n_outputs)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = F.leaky_relu(call_bn(self.bn1, h), 0.01)
        h = self.c2(h)
        h = F.leaky_relu(call_bn(self.bn2, h), 0.01)
        h = self.c3(h)
        h = F.leaky_relu(call_bn(self.bn3, h), 0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c4(h)
        h = F.leaky_relu(call_bn(self.bn4, h), 0.01)
        h = self.c5(h)
        h = F.leaky_relu(call_bn(self.bn5, h), 0.01)
        h = self.c6(h)
        h = F.leaky_relu(call_bn(self.bn6, h), 0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c7(h)
        h = F.leaky_relu(call_bn(self.bn7, h), 0.01)
        h = self.c8(h)
        h = F.leaky_relu(call_bn(self.bn8, h), 0.01)
        h = self.c9(h)
        h = F.leaky_relu(call_bn(self.bn9, h), 0.01)
        # In the final average pooling, we will reduce one dimension of the h
        h = F.avg_pool2d(h, kernel_size=h.data.shape[2])
        # Remove the last dimension [[[x]]] -> [[x]]
        # h.view will share the same memory with h
        # faster than copy()
        h = h.view(h.size(0), h.size(1))
        # Last linear layer to generate final output
        logit = self.l_c1(h)
        # Not sure what is bn_c1
        # if self.top_bn:
        #     logit = call_bn(self.bn_c1, logit)
        return logit

        