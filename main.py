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
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )
    test_dataset = CIFAR100(root='./data/',
                                download=True,
                                train=False,
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )


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
    batch_size = target.size(0)

    # pred is the indices of top maxk values in output
    _, pred = output.topk(maxk, 1, True, True)
    # This step is kind of meaningless
    # If pred is in form of [...var...], this make no sense
    # because [...var...].t() will remain the same shape
    # However, if pred is in form of [[...var...]], it will be
    # transposed to [...[],[],[]...]
    # pred = [[pred[i]] for i in range(len(pred))] this only work for nparray
    # We can use:
    # pred = pred.expand(1, -1).t()
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        # k is %, multiply 100 to get the #correct
        res.append(correct_k.mul_(100.0/batch_size))
    
    return res

def train(train_loader, epoch, model1, optimizer1, model2, optimizer2):
    print("Training {}...".format(model_str))
    pure_ratio_list = []
    pure_ratio_list1 = []
    pure_ratio_list2 = []

    train_total = 0
    train_correct = 0
    train_total2 = 0
    train_correct2 = 0

    ### Only for test ####
    # for a, b in enumerate(train_loader):
    #     print(a)
    #     print(len(b))
    #     break


    for i, (images, labels, indexes) in enumerate(train_loader):
        #
        #
        # Check if there's relationship between indexes and i
        #
        #
        ind = indexes.cpu().numpy().transpose()

        ####### TEST ##########
        # print(indexes)
        #######################

        if i > args.num_iter_per_epoch:
            break
        
        # Load data into GPU
        # Instead of using Variable(), we can just assign requires_grad=True when declair Tensor
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Logits is the unnormalised output(prediction) of the model
        logits1 = model1(images)
        prec1, _ = accuracy(logits1, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec1

        logits2 = model2(images)
        prec2, _ = accuracy(logits2, labels, topk=(1, 5))
        train_total2 += 1
        train_correct2 += prec2

        loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not)
        pure_ratio_list1.append(pure_ratio_1*100)
        pure_ratio_list2.append(pure_ratio_2*100)

        # Calculate the gradient of loss_1(2) by back propagation
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

        ######### TEST ############
        # print(type(loss_1.data.item()))
        # print(loss_2.data.item())

        # print(prec1.data.item())
        ###########################

        if (i+1) % args.print_freq == 0:
            print('Epoch [{:d}/{:d}], Iter [{:d}/{:d}], Training Accuracy1: {:.4f}, Training Accuracy2: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Pure Ratio1: {:.4f}, Pure Ratio2: {:.4f}'.format(epoch+1,
                                                                                                                 args.n_epoch, i+1, len(train_dataset)//batch_size, 
                                                                                                                 prec1.data.item(), prec2.data.item(),
                                                                                                                 loss_1.data.item(),
                                                                                                                 loss_2.data.item(),
                                                                                                                 np.sum(pure_ratio_list1)/len(pure_ratio_list1),
                                                                                                                 np.sum(pure_ratio_list2)/len(pure_ratio_list2)))
        
    train_acc1 = float(train_correct)/float(train_total)
    train_acc2 = float(train_correct2)/float(train_total2)

    return train_acc1, train_acc2, pure_ratio_list1, pure_ratio_list2


def evaluate(test_loader, model1, model2):
    print('Evaluating {}...'.format(model_str))
    model1.eval()
    correct1 = 0
    total1 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits1 = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()
    
    model2.eval()
    correct2 = 0
    total2 = 0
    for image, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits2 = model2(images)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()

    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)

    return acc1, acc2

def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    # drop_last=True will drop the last incomplete batch
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=batch_size,
                                                num_workers=args.num_workers,
                                                drop_last=True,
                                                shuffle=False)

    ##### Only for test ####
    # for a, b in enumerate(train_loader):
    #     print(a)
    #     print(len(b))
    # return

    # Define models
    print('building model...')
    cnn1 = CNN(input_channel=input_channel, n_outputs=num_classes)
    cnn1.cuda()
    print(cnn1.parameters)
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)
    
    cnn2 = CNN(input_channel=input_channel, n_outputs=num_classes)
    cnn2.cuda()
    print(cnn2.parameters)
    optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)

    mean_pure_ratio1=0
    mean_pure_ratio2=0

    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc1 train_acc2 test_acc1 test_acc2 pure_ratio1 pure_ratio2\n')

    epoch=0
    train_acc1=0
    train_acc2=0
    # evaluate models with random weights
    test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
    print('Epoch [{:d}/{:d}] Test Accuracy on the {} test images: Model1 {:.4f}% Model2 {:.4f}% Pure Ratio1 {:.4f}% Pure Ratio2 {:.4f}%'.format(epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' '  + str(mean_pure_ratio1) + ' '  + str(mean_pure_ratio2) + "\n")

    # training
    for epoch in range(1, args.n_epoch):
        # train models
        cnn1.train()
        adjust_learning_rate(optimizer1, epoch)
        cnn2.train()
        adjust_learning_rate(optimizer2, epoch)
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list=train(train_loader, epoch, cnn1, optimizer1, cnn2, optimizer2)
        # evaluate models
        test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
        # save results
        mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
        print('Epoch [{:d}/{:d}] Test Accuracy on the {} test images: Model1 {:.4f}% Model2 {:.4f}%, Pure Ratio 1 {:.4f}%, Pure Ratio 2 {:.4f}%'.format(epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' ' + str(mean_pure_ratio1) + ' ' + str(mean_pure_ratio2) + "\n")
    pass

if __name__ == '__main__':
    main()
