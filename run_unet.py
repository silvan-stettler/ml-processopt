# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:12:45 2019

@author: silus
"""

import torch
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from unet.unet_helpers import MicroscopeImageDataset, ToTensor
from unet.unet import UNet

etching_dataset = MicroscopeImageDataset(img_dir='./images/etching/', mask_dir='./images/etching/masks/',
                                         transform=ToTensor(), 
                                        split_samples=4)

# Define U-Net
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_classes=2, padding=True, up_mode='upsample').to(device)
optim = torch.optim.Adam(model.parameters())
dataloader = DataLoader(etching_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
epochs = 10

for _ in range(epochs):
    print("Epoch {0}".format(_))
    for smple in dataloader:
        X = smple['image'].to(device)  # [N, 1, H, W]
        y = smple['mask'].to(device)  # [N, H, W] with class indices (0, 1)
        prediction = model(X)  # [N, 2, H, W]
        loss = F.cross_entropy(prediction, y)

        optim.zero_grad()
        loss.backward()
        optim.step()