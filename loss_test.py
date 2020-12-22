import torch.nn as nn
import torch
import numpy as np
from math import *

target = torch.tensor([[1,0,1],[0,1,1]]).float()
raw_output = torch.randn(2,3)
output = torch.sigmoid(raw_output)
print(output)

result = np.zeros((2,3), dtype=np.float)
for ix in range(2):
    for iy in range(3):
        if(target[ix, iy]==1): result[ix, iy] += -log(output[ix, iy])
        elif(target[ix, iy]==0): result[ix, iy] += -log(1-output[ix, iy])

print(result)
print(np.mean(result))

loss_fn = torch.nn.BCELoss(reduction='none')
print(loss_fn(output, target))
loss_fn = torch.nn.BCELoss(reduction='mean')
print(loss_fn(output, target))
loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
print(loss_fn(raw_output, target))

