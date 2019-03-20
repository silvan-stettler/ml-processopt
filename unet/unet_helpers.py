# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 23:29:24 2019

@author: silus
"""
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join

SAVE_PATH = '../trained/unet.pt'
class MicroscopeImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, split_samples=None):
        self.root_dir = img_dir
        self.mask_dir = mask_dir
        self.file_list = [ f for f in listdir(self.root_dir) if isfile(join(self.root_dir,f)) ]
        self.transform = transform
        if split_samples:
            assert isinstance(split_samples, (int,tuple))
        self.split_samples = split_samples
    def __len__(self):
        if self.split_samples:
            if isinstance(self.split_samples, int):
                k = self.split_samples**2
            else:
                k = self.split_samples[0]*self.split_samples[1]
        else:
            k = 1
        return len(self.file_list)*k
    def __getitem__(self, index):
        idx = index
        if self.split_samples:
            if isinstance(self.split_samples, int):
                k = self.split_samples**2
            else:
                k = self.split_samples[0]*self.split_samples[1]
            sub_idx = int(index%k)
            idx = int((index-sub_idx)/k)
        img_name = join(self.root_dir,self.file_list[idx])
        mask_name = join(self.mask_dir, self.file_list[idx])[:-4] + '_mask.png'
        image = io.imread(img_name)
        mask = io.imread(mask_name)
        mask[mask <= 127] = 0
        mask[mask > 127] = 1
        sample = {'image':image,'mask':mask}
        if self.split_samples:
            sample = self.split_sample_(sample, sub_idx)
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def split_sample_(self,sample, n):
        "Divide images into a bunch of subimages"
        
        img, msk = sample['image'], sample['mask']
        assert img.shape == msk.shape
        h, w = img.shape[:2]
        if isinstance(self.split_samples, int):
            h_int = np.linspace(0, h, num=self.split_samples+1, dtype=np.uint32)
            w_int = np.linspace(0, w, num=self.split_samples+1, dtype=np.uint32)
            nh = n%self.split_samples
            nw = (n-nh)/self.split_samples
        else:
            h_int = np.linspace(0, h, num=self.split_samples[0]+1, dtype=np.uint32)
            w_int = np.linspace(0, w, num=self.split_samples[1]+1, dtype=np.uint32)
            nh = n % self.split_samples[0]
            nw = (n-nh)/self.split_samples[0]
        h0, h1 = h_int[int(nh)], h_int[int(nh+1)]
        w0, w1 = w_int[int(nw)], w_int[int(nw+1)]
        return {'image':img[h0:h1, w0:w1], 'mask':msk[h0:h1, w0:w1]}
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        assert image.shape == mask.shape
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        msk = transform.resize(mask, (new_h, new_w))

        return {'image':img,'mask':msk}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        assert image.shape == mask.shape

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        mask = mask[top: top + new_h,
                      left: left + new_w]

        return {'image':image,'mask':mask}

class BrightnessContrastAdjustment(object):
    
    def __init__(self, adjustment, order):
        assert isinstance(adjustment, tuple)
        self.brightn_corr = adjustment[0]
        self.contr_corr = adjustment[1]
        
        assert order in ['contrast','brightness']
        self.op_order = order
    
    def __call__(self, sample):
        out = sample['image']
        assert len(sample['image'].shape) >1
        if len(sample['image'].shape)>2:
            assert sample['image'].shape[2]==1
            out = sample['image'].reshape(sample.shape[:2])
        
        if self.op_order=='contrast':
            out = np.clip(self.contr_corr*out + self.brightn_corr,0,255).astype(np.uint8)
        elif self.op_order=='brightness':
            out = np.clip(self.contr_corr*(out + self.brightn_corr),0,255).astype(np.uint8)
        return {'image':out,'mask':sample['mask']}    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image,mask = sample['image'], sample['mask']
        assert image.shape == mask.shape
        assert len(image.shape) == 2 or (len(image.shape)==3 and image.shape[2]==1)
        
        if len(image.shape) == 2:
            image = torch.from_numpy(image.transpose()).view(1,image.shape[1],image.shape[0]).type(torch.FloatTensor)
            mask = torch.from_numpy(mask.transpose()).type(torch.LongTensor)
        
        # TODO: XXXXXXXXXXXXXXXXXXXX when len(shape) isn't 2
        return {'image':image,
                'mask': mask}
        
def train_unet(model, device, optimizer, criterion, dataloader, 
               epochs=10, lambda_=1e-3, reg_type=None, save=False):
    
    avg_epoch_loss = []
    for _ in range(epochs):
        print("Epoch {0}".format(_))
        loss_accum = 0
        for smple in dataloader:
            X = smple['image'].to(device)  # [N, 1, H, W]
            y = smple['mask'].to(device)  # [N, H, W] with class indices (0, 1)
            
            mu, std = X.mean(), X.std()
            X.sub_(mu).div_(std)
            
            prediction = model(X)  # [N, 2, H, W]
            loss = criterion(prediction, y)
            
            if reg_type:
                assert reg_type in ['l2','l1']
                if reg_type == 'l2':
                    for p in model.parameters():
                        loss += lambda_ * p.pow(2).sum()
                    
            print("Cross entropy loss: {:.02f}".format(loss.item()))
            loss_accum += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if reg_type=='l1':
                with torch.no_grad():
                    for p in model.parameters():
                        p.sub_(p.sign() * p.abs().clamp(max = lambda_))
                        
        avg_epoch_loss.append(loss_accum/len(dataloader))
    
    if save:
        torch.save(model.state_dict(),SAVE_PATH)
