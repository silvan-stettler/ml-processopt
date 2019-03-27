# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:41:21 2019

@author: silus
"""

import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

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

def infere_model(model, X):
    " Test the trained model on a sample"
    assert isinstance(X, torch.FloatTensor)
    assert X.shape[0] == 2
    
    X.unsqueeze_(0)
    model.eval()
    output = model(X)
    output.squeeze_(0)
    output = -torch.nn.functional.log_softmax(output, dim=0)
    output_labels = output.argmax(dim=0)
    output_labels.unsqueeze_(0)
    print(output_labels.shape)
    
    return output_labels.numpy()
    #output_labels = output_labels.permute(1,2,0)
    #img = transforms.functional.to_pil_image(np.uint8(output_labels))
    
    #return img
    
def visualize_sample(fig, sample, suppl_image=None):
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
    
    elements.append(mask)
    
    if suppl_image is not None:
        assert len(suppl_image.shape) <= 4
        if len(suppl_image.shape) == 3:
            for img in suppl_image:
                elements.append(img)
        else:
            elements.append(suppl_image)
        
    n_elements = len(elements)
    
    axes = []
    for i,e in enumerate(elements):
        ax = fig.add_subplot(1,n_elements,i+1)
        ax.imshow(e, cmap='gray', vmin=e.min(), vmax=e.max())
        axes.append(ax)
        
    return axes
        
        
    
    