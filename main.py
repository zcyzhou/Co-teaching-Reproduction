# -*- coding:utf-8 -*-#
import os
import argparse, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
import datetime
import shutil

"""
The CIFAR and MNIST datasets can be imported directly from the torchvision now
"""
# from torchvision.datasets.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from data.cifar import CIFAR10, CIFAR100
from model import CNN
from loss import loss_coteaching

"""
Crearte parser for the command line arguments
"""
parser = argparse.ArgumentParser()
# learning rate
parser.add_argument('--lr', type=float, default=0.001)
# result directory
parser.add_argument('--result_dir', type=str, default='results/', help='dir to save result files')
# noise rate
parser.add_argument('--noise_rate', type=float, default=0.2, help='corruption rate, should be less than 1')
# forget rate
parser.add_argument('--forget_rate', type=float, default=None, help='forget rate')
# noise_type
parser.add_argument('--noise_type', type=str, default='pairflip', help='[pairflip, symmetric]')
parser.add_argument('--num_gradual', type=int, default=10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
# As described in Python doc, the default value of 'store_true' is False
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--dataset', type=str, default='mnist', help='mnist, cifar10, cifar100')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)

args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper parameters
batch_size = 128
learning_rate = args.lr

"""
LOAD DATASET
1. mnist
2. cifar10
3. cifar100
There are several default values:
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
"""

# mnist
if args.dataset == 'mnist':
    # What is input_channel ????
    input_channel = 1
    num_classes = 10
    train_dataset = MNIST(root = './data/', \
                            download = True, \
                            train = True, \
                            transform = transforms.ToTensor(), \
                            noise_type = args.noise_type, \
                            noise_rate = args.noise_rate \
                            )
    test_dataset = MNIST(root='./data/', \
                            download=True, \
                            train=False, \
                            transform=transforms.ToTensor(), \
                            noise_type=args.noise_type, \
                            noise_rate=args.noise_rate \
                            )

# cifar10
if args.dataset == 'cifar10':
    input_channel = 3
    num_classes = 10
    train_dataset = CIFAR10(root='./data/',
                                download=True,
                                train=True,
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )
    test_dataset = CIFAR10(root='./data/',
                                download=True,
                                train=False,
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )

# cifar100
if args.dataset == 'cifar100':
    input_channel = 3
    num_classes = 100
    args.top_bn = False
    args.epoch_decay_start = 100
    args.n_epoch = 200

    train_dataset = CIFAR100(root='./data/',
                                download=True,
                                train=True,
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type=,
                                noise_rate=args.noise_rate
                                )
    test_dataset = CIFAR100(root='./data/',
                                download=True,
                                train=False,
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )

"""
@ Question:
Need to figure out what is the function of forget_rate
"""
if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate

noise_or_not = train_dataset.noise_or_not

"""
The following part is for adjusting learning rate & betas
for Adam Optimizer
"""
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch

for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)

# define drop rate schedule
rate_schedule = np.ones(args.n_epoch) * forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)

"""
From the commment block above to here, do not really understand the code
"""

"""
Following part is for storing the result
"""
save_dir = args.result_dir + '/' + args.dataset + '/coteaching/'
# if Not exist then create the dir
if not os.path.exists(save_dir):
    os.system('mkdir -p {}'.format(save_dir))

model_str = args.dataset + '_coteaching_' + args.noise_type + '_' + str(args.noise_rate)
txtfile = save_dir + '/' + model_str + '.txt'
now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv {} {}'.format(txtfile, txtfile + '.bak-{}'.format(now_time)))


# Function to evaluate the accuracy
def accuracy(logit, target, topk=(1,)):
    """Compute the precision at k for a specified k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    pass

def train():
    pass

def evaluate():
    pass

def main():
    pass

if __name__ == '__main__':
    main()