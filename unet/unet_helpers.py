# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 23:29:24 2019

@author: silus
"""
import torch
from skimage import io, transform
import numpy as np
import time

from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
from .utils import label_mask

SAVE_PATH = '../trained/unet.pt'
class MicroscopeImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, mask_label_info, read_top=True, split_samples=None, transf=None):
        self.bottom_dir = img_dir[0]
        self.top_dir = img_dir[1]
        self.mask_dir = mask_dir
        self.read_top = read_top
        self.transform_samples = transf
        self.mask_label_info = mask_label_info
        self.file_list = [ f for f in listdir(self.bottom_dir) if isfile(join(self.bottom_dir,f)) ]
        
        # Return top and bottom image by default
        self.return_channel = None
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
        img_name = join(self.bottom_dir,self.file_list[idx])
        if self.mask_dir is not None:
            mask_name = join(self.mask_dir, self.file_list[idx])[:-4] + '_mask.png'
            mask = io.imread(mask_name, as_gray=True)
            mask = label_mask(mask, self.mask_label_info)
        else:
            mask = None
        
        # Merge top and bottom picture to single input
        if self.read_top:
            img_name_top = join(self.top_dir, self.file_list[idx])[:-4] + '_top.png'
            image_col = io.ImageCollection([img_name, img_name_top])
            # 3D image
            image = io.concatenate_images(image_col)
            image = image.transpose((1,2,0))/image.max()
        else:
            # Force 3D even if just a one channel image
            image = io.imread(img_name)[:,:, np.newaxis]
            image = image/image.max()
        
        sample = {'image': image, 'mask': mask}
        
        if self.split_samples:
            sample = self.split_sample_(sample, sub_idx)
            
        if self.transform_samples:
            sample = self.transform_samples(sample)
            
        if self.return_channel is not None:
            sample['image'] = sample['image'][self.return_channel:self.return_channel+1]
            
        return sample
    
    def split_sample_(self,sample, n):
        "Divide images into a bunch of subimages"
        
        img, msk = sample['image'], sample['mask']
        assert img.shape[:2] == msk.shape
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
    
    def infer(self, idx, model, use_cuda=False):
        " Test the trained model on a sample"
    
        sample = self.__getitem__(idx)
        
        if use_cuda and torch.cuda.is_available():
            X = X.cuda()
        X = sample['image']
        assert isinstance(X, torch.FloatTensor)
        
        X.unsqueeze_(0)
        model.eval()
        output = model(X)
        output.squeeze_(0)
        output = -torch.nn.functional.log_softmax(output, dim=0)
        output_labels = output.argmax(dim=0)
        output_labels.unsqueeze_(0)
        print(output_labels.shape)
        
        return sample, output_labels.numpy()
    
    def bottom(self):
        self.return_channel = 0
    
    def top(self):
        if not self.read_top:
            print("Top pictures not available. Initialize with 'read_top = True'")
            self.return_channel = 0
        else:
            self.return_channel = 1
    

class ConcatDatasets(MicroscopeImageDataset):
    def __init__(self, *datasets):
        # Initialize superclass ??
        self.datasets = datasets
        self.nbr_samples = [len(d) for d in self.datasets]
    def __getitem__(self, idx): 
        cum_samples = np.cumsum(self.nbr_samples)
        idx_larger = np.where(cum_samples > idx)
        dset = self.datasets[min(idx_larger[0])]
    
        if min(idx_larger[0]) == 0:
            idx_inside_dset = idx
        else:
            idx_inside_dset = idx - cum_samples[min(idx_larger[0])-1]
      
        return dset[idx_inside_dset]
    
    def __len__(self):
        return np.sum(self.nbr_samples)
    
    def bottom(self):
        for dset in self.datasets:
            dset.bottom()
    
    def top(self):
        for dset in self.datasets:
            dset.top()
    
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
        if mask is not None:
            assert image.shape[:2] == mask.shape
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        if mask is not None:
            msk = transform.resize(mask, (new_h, new_w), preserve_range=True)
        else: 
            msk = None
        
        return {'image':img,'mask':msk}

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        img = transform.rotate(image, self.angle, preserve_range=True, mode='edge')
        msk = transform.rotate(mask, self.angle, preserve_range=True, mode='edge')
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
        self.brightn_corr, self.contr_corr = adjustment
        
        assert order in ['contrast','brightness']
        self.op_order = order
    
    def __call__(self, sample):
        out = sample['image']
        
        if self.op_order=='contrast':
            out = np.clip(self.contr_corr*out + self.brightn_corr,0,1)
        elif self.op_order=='brightness':
            out = np.clip(self.contr_corr*(out + self.brightn_corr),0,1)
        return {'image':out,'mask':sample['mask']}    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        image,mask = sample['image'], sample['mask']
        
        image = image.transpose((2, 0, 1))
        
        if mask is not None:
            msk = torch.from_numpy(mask).type(torch.LongTensor)
        else:
            msk = None
        
        return {'image':torch.from_numpy(image).type(torch.FloatTensor),
                'mask': msk}
        
def train_unet(model, optimizer, criterion, dataloader, 
               epochs=10, lambda_=1e-3, reg_type=None, use_cuda=False):
    
    avg_epoch_loss = []
    for _ in range(epochs):
        print("Epoch {0}".format(_))
        start = time.time()
        loss_accum = 0
        for i,smple in enumerate(dataloader):
            X = smple['image']  # [N, 1, H, W]
            y = smple['mask']  # [N, H, W] with class indices (0, 1)
            
            if use_cuda and torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            
            # Normalization is done with 2D batch norm labels in UNet
            
            prediction = model(X)  # [N, 2, H, W]
            loss = criterion(prediction, y)
            
            if reg_type:
                assert reg_type in ['l2','l1']
                if reg_type == 'l2':
                    for p in model.parameters():
                        loss += lambda_ * p.pow(2).sum()
            loss_accum += loss.item()
            
            #if (i%10 == 0) & (i != 0):        
            #    print("Batch {:d}, Cross entropy loss: {:.02f}".format(i,loss_accum/i))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if reg_type=='l1':
                with torch.no_grad():
                    for p in model.parameters():
                        p.sub_(p.sign() * p.abs().clamp(max = lambda_))
                        
        avg_epoch_loss.append(loss_accum/len(dataloader))
        end = time.time()-start
        print(criterion.__class__.__name__ + ": {0:.03f} Duration: {1:.02f}".format(loss_accum/len(dataloader), end))
    
    return avg_epoch_loss, model

def imread_convert(f):
    return io.imread(f, as_gray=True)