# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:41:21 2019

@author: silus
"""

import numpy as np

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