from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    """Function for computing the loss of the model

    Args:
        y_1: logist of model1
        y_2: logist of model2
        t: target labels
        ind: 
    """
    # reduction='none' will return cross_entropy loss for each one
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    # Interesting operation to sort the loss_1
    ind_1_sorted = np.argsort(loss_1.data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = np.argsort(loss_2.data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    # noise_or_not is an array of [True, False]
    # np.sum will return the number of True
    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2
