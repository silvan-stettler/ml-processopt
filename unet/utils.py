# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:41:21 2019

@author: silus
"""

import numpy as np

def label_mask(mask_img, labels, thresh):
    
    assert isinstance(mask_img, np.ndarray)
    assert len(labels)-1 == len(thresh)
    
    out = mask_img
    
    for i,l in enumerate(labels):
        if i == 0:
            out[out <= thresh[i]] = l
        elif i== len(thresh):
            out[out > thresh[i-1]] = l
        else:
            out[out <= thresh[i] and out > thresh[i-1]] = l
        
    return out