import torch
from torch import nn
from torch.nn import functional as F

def my_ce_loss(output, target):
    A = torch.log(torch.sum(torch.exp(output), dim=1))    
    B = torch.sum(F.one_hot(target, num_classes=5).float()*output, dim=1)
    
    return torch.mean(A-B)

def my_mse_loss(output, target):
    A = output
    B = F.one_hot(target, num_classes=5).float()
    
    return torch.mean((A-B)**2)

def my_ce_mse_loss(output, target):
    A = torch.log(torch.sum(torch.exp(output), dim=1))    
    B = torch.sum(F.one_hot(target, num_classes=5).float()*output, dim=1)

    C = output
    D = F.one_hot(target, num_classes=5).float()
    
    softmax = nn.Softmax(dim=1)
    softmax_output = softmax(output)
    
    return torch.mean(A-B) + torch.mean(softmax_output*((C-D)**2))

""" output = torch.Tensor([[0.3982, 0.8125, 0.6213, 0.9323, 0.5141],
                       [0.3122, 0.2135, 0.1213, 0.9123, 0.3241],
                       [0.2352, 0.1234, 0.7341, 0.1235, 0.3783]])

target = torch.LongTensor([0, 2, 1])
# target = F.one_hot(target, num_classes=5).float()

# criterion1 = nn.CrossEntropyLoss()
criterion1 = my_ce_loss
# criterion2 = nn.MSELoss()
criterion2 = my_mse_loss
criterion3 = my_ce_mse_loss

loss1 = criterion1(output, target)
loss2 = criterion2(output, target)
loss3 = criterion3(output, target)

print(loss1)
print(loss2)
print(loss3) """