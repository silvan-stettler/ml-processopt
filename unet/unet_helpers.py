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
from .utils import label_mask, find_closest_pixel

SAVE_PATH = '../trained/unet.pt'
PIX_WEIGHT_PATH='./pixw'
class MicroscopeImageDataset(Dataset):
    def __init__(self,name , img_dir, mask_dir, mask_label_info, read_top=True, 
                 split_samples=None, transf=None, pixel_weights=(1,5), recompute=False):
        self.name = name
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
        
        # Compute class frequencies
        self.compute_class_freq_()
        self.compute_pixel_weight_(pixel_weights, recompute=recompute)
        self.pixel_weights = True
        
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
            
        if hasattr(self, 'pixel_weights') and hasattr(self, 'label_weights'):
            sample = self.get_pixel_weight_(sample, index)
            
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
        X = sample['image']
        assert isinstance(X, torch.FloatTensor)
        
        if use_cuda and torch.cuda.is_available():
            X = X.cuda()
        
        X.unsqueeze_(0)
        model.eval()
        output = model(X)
        output.squeeze_(0)
        output = torch.nn.functional.log_softmax(output, dim=0)
        output_labels = output.argmax(dim=0)
        output_labels.unsqueeze_(0)
        
        return sample, output_labels.cpu().numpy()
    
    def bottom(self):
        self.return_channel = 0
    
    def top(self):
        if not self.read_top:
            print("Top pictures not available. Initialize with 'read_top = True'")
            self.return_channel = 0
        else:
            self.return_channel = 1
    
    def compute_class_freq_(self):
        
        all_labels = torch.Tensor(self.mask_label_info[0])
        tot_freq = torch.zeros_like(all_labels)
        for i in range(self.__len__()):
            sample = self.__getitem__(i)
            labels = sample['mask']
            
            
            unique_labels, freq = torch.unique(labels, return_counts=True)
            l_freq = torch.zeros_like(tot_freq)
            
            for l,f in zip(unique_labels, freq):
                idx = (all_labels==l.item()).nonzero().item()
                l_freq[idx] = f.item()
            
            tot_freq += l_freq
                
        print("Class frequecies:")
        for i,l in enumerate(all_labels):
            print("{0} {1:.02f}% of pixels".format(l.item(), 100*tot_freq[i].item()/tot_freq.sum().item()))
        
        self.frequencies = tot_freq
        self.label_weights = (all_labels, tot_freq.sum()/tot_freq)
    
    def compute_pixel_weight_(self, params, recompute=False):
        """
            Computes loss weight of a pixel as a function of class frequency and
            distance to the closest pixel of another class
        """
        
        for i in range(self.__len__()):
            sample = self.__getitem__(i)
            
            filename = 'w_'+self.name+'_'+str(i)+'.pt'
            
            if filename not in listdir(PIX_WEIGHT_PATH) or recompute:
                labels = sample['mask']
                label_weights = self.label_weights[1]
                freq_weight = torch.zeros_like(labels, dtype=torch.float)
                
                # Assign weight based on class of a pixel 
                for j,l in enumerate(self.label_weights[0]):
                    freq_weight[labels==l.item()] = label_weights[j]
        
                if params is not None:
                    assert isinstance(params, tuple)
                    d_weight_multiplier, sigma = params
                    distances = find_closest_pixel(labels, self.label_weights[0])
                    dist_weight = torch.exp(-distances.pow(2)/(2*sigma**2))
                    
                    weight = freq_weight + label_weights.mean() * d_weight_multiplier * dist_weight
                else:
                    weight = freq_weight
                    
                torch.save(weight, PIX_WEIGHT_PATH+'/'+filename)
        
    def get_pixel_weight_(self, sample, idx):
        filename = 'w_'+self.name+'_'+str(idx)+'.pt'
        try:
            weight = torch.load(PIX_WEIGHT_PATH+'/'+filename)
            return {'image':sample['image'], 'mask': sample['mask'], 'weight': weight}
        except FileNotFoundError:
            print("Pixel weights {0} not found for sample index {1}".format(filename, idx))
    
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
        if mask is not None:
            assert image.shape[:2] == mask.shape

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
        
def init_weights(module):
    """
        Initialize the weights of module.
        Kaiming initialization for Conv and Linear layers similar to the 
        original publication on U-Net
    """
    
    if isinstance(module, torch.nn.BatchNorm2d):
        module.reset_parameters()
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d) or isinstance(module, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight.data, nonlinearity='relu')
        
        
def train_unet(model, optimizer, criterion, dataloader, 
               epochs=10, lambda_=1e-3, reg_type=None, use_cuda=False, pixel_weights=True):
    
    model.apply(init_weights)
    model.train()
    
    avg_epoch_loss = []
    for _ in range(epochs):
        print("Epoch {0}".format(_))
        start = time.time()
        loss_accum = 0
        for i,smple in enumerate(dataloader):
            X = smple['image']  # [N, 1, H, W]
            y = smple['mask']  # [N, H, W] with class indices (0, 1)
            
            if pixel_weights:
                w = smple['weight']
            if use_cuda and torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
                if pixel_weights:
                    w = w.cuda()
            # Normalization is done with 2D batch norm labels in UNet
            
            prediction = model(X)  # [N, 2, H, W]
            
            if pixel_weights:
                loss = criterion(prediction, y, w)
            else:
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
    
