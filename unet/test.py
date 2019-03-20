# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:41:21 2019

@author: silus
"""

import torch
import torch.nn.functional as F

f = torch.tensor([[-1., -3., 4.], [-3., 3., -1.]])
target = torch.tensor([0, 1])

loss = F.cross_entropy(f, target)
