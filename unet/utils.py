# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:41:21 2019

@author: silus
"""
import torch
import torch.nn.functional as F
import numpy as np
import scipy.spatial as sp
from itertools import permutations

def count_module_train_params(module):
    
    assert hasattr(module, 'parameters')
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def conv2d_out_shape(shape, out_channels, kernel_size=3, padding=0, stride=1, dilation=1, conv_type='down'):
    """
    Calculates the shape of the output after a convolutional layer, eg. Conv2d 
    or a max pooling layer
    Args:
        shape       Shape of the input to the layer in format C x H x W. Can be 
                    a tuple, list, torch.Tensor or torch.Size
        out_channels    Number of channels of the ouput
    Returns:
        out_shape   List with three elements [C_out, H_out, W_out]
    """
    if not isinstance(kernel_size, torch.Tensor):
        kernel_size = torch.Tensor((kernel_size, kernel_size))
    if not isinstance(stride, torch.Tensor):
        stride = torch.Tensor((stride, stride))
    
    # Handle different input types
    if isinstance(shape, torch.Size):
        chw = torch.Tensor([s for s in shape])
    elif isinstance(shape, torch.Tensor):
        chw = shape
    else:
        chw = torch.Tensor(shape)
    
    out_shape = chw
    
    if conv_type is 'down':
        out_shape[1:3] = torch.floor((chw[1:3] + 2*padding - dilation*(kernel_size-1)-1)/stride + 1)
    elif conv_type is 'up':
        out_shape[1:3] = (chw[1:3]-1)*stride - 2*padding + kernel_size
    else:
        print("Unknown convolution type")
    out_shape[0] = out_channels
    
    # return as list
    
    return [int(s.item()) for s in out_shape]


def label_mask(mask_img, label_info):
    
    assert isinstance(mask_img, np.ndarray)
    labels = label_info[0]
    thresh = label_info[1]
    assert len(labels)-1 == len(thresh)
    
    out = np.zeros_like(mask_img)
    
    for i,l in enumerate(labels):
        if i == 0:
            out[mask_img <= thresh[i]] = l
        elif i== len(thresh):
            out[mask_img > thresh[i-1]] = l
        else:
            out[(mask_img <= thresh[i]) & (mask_img > thresh[i-1])] = l
        
    return out
    
def visualize_sample(fig, sample, suppl_image=None, cmaps=None):
    try:
        image, mask = sample['image'], sample['mask']
    except KeyError:
        print("Wrong sample format")
        return None
    
    if len(image.shape) > 3:
        image.squeeze_(0)
        
    elements = []
    for layer in image:
        elements.append(layer)
    
    if mask is not None:
        elements.append(mask)
    
    if suppl_image is not None:
        assert len(suppl_image.shape) <= 4
        if len(suppl_image.shape) == 3:
            for img in suppl_image:
                elements.append(img)
        else:
            elements.append(suppl_image)
        
    n_elements = len(elements)
    
    if cmaps is not None:
        assert len(cmaps) == len(elements)
    axes = []
    for i,e in enumerate(elements):
        ax = fig.add_subplot(1,n_elements,i+1)
        if cmaps is not None:
            cmap = cmaps[i]
            if cmap == '':
                cmap = 'gray'
        else:
            cmap = 'gray'
        ax.imshow(e, cmap=cmap, vmin=e.min(), vmax=e.max())
        axes.append(ax)
        
    return axes
        
def find_closest_pixel(image, labels):
    
    indices = {}
    for i,l in enumerate(labels):
        loc = (image == l.item())
        if not loc.any():
            continue
        indices[str(l.item())] = (loc, loc.nonzero(), sp.KDTree(loc.nonzero().numpy(), leafsize=100))
        
    dist = torch.zeros_like(image, dtype=torch.float)
    add_dist = torch.zeros_like(image, dtype=torch.float)
    
    if len(indices.keys()) < 2:
        dist.fill_(50)
    else:
        for kcomb in permutations(indices.keys()):
            pixel = indices[kcomb[0]]
            target = indices[kcomb[1]]

            d,_ = target[2].query(pixel[1].numpy())
            add_dist[pixel[0]] = torch.Tensor(d)
            dist += add_dist
    
    return dist

def pixel_cross_entropy(pred, target, weights=None):
    """
        Cross entropy loss where a weight is assigned for the loss of each pixel
    """
    batch_size, num_classes, h, w = pred.shape
    logits = F.log_softmax(pred, 1).permute(0,2,3,1)
    
    if target.is_cuda:
        device = target.data.get_device()
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, h, w, 1)
                                               .cuda(device)
                                               .eq(target.data.unsqueeze(3).repeat(1,1,1,num_classes)))
        wghts = weights.unsqueeze(0).cuda(device)
    else:
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, h, w, 1)
                                               .eq(target.data.unsqueeze(3).repeat(1,1,1,num_classes)))
        wghts = weights.unsqueeze(0)  
         
    loss = -logits.masked_select(one_hot_mask).view(batch_size, h, w) * wghts 
    
    return loss.mean((1,2))

class PixelWeightCrossEntropyLoss(torch.nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, aggregate='mean'):
        super(PixelWeightCrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate

    def forward(self, inp, target, weights=None):
        if self.aggregate == 'sum':
            return pixel_cross_entropy(inp, target, weights).sum()
        elif self.aggregate == 'mean':
            return pixel_cross_entropy(inp, target, weights).mean()
        elif self.aggregate is None:
            return pixel_cross_entropy(inp, target, weights)
    
def add_to_summary(summary, layer, in_shape, out_shape, n_param):
    summary['Trainable params'].append(n_param)
    summary['Layer'].append(layer)
    summary['Input shape'].append(str(in_shape))
    summary['Output shape'].append(str(out_shape))      
    
    
