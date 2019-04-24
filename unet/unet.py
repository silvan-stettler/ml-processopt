# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 23:20:22 2019

@author: silus
"""

import torch
from torch import nn
import torch.nn.functional as F
from .utils import count_module_train_params, conv2d_out_shape, add_to_summary

class UNet(nn.Module):
    def __init__(self, in_shape, n_classes=2, depth=5, wf=6, padding=False,
                 kernel_size=3, batch_norm=False, pooling=2, up_mode='upconv', 
                 name=None):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_shape = in_shape
        self.down_path = nn.ModuleList()
        self.module_summary = []
        
        for i in range(depth):
            add_block = UNetConvBlock(prev_shape, 2**(wf+i),
                                      padding, batch_norm, kernel_size=kernel_size)
            self.down_path.append(add_block)
            
            # Add convolution block to model summary
            block_summary = add_block.block_summary 
            prev_shape = add_block.outp_shape
            
            # Pooling layer (as functional in forward pass)
            if i != depth-1:
                pool_inp_shape = prev_shape
                prev_shape = conv2d_out_shape(prev_shape, prev_shape[0], kernel_size=2, stride=2)
                add_to_summary(block_summary, 'maxpool_2', pool_inp_shape, prev_shape, 0)
            
            block_summary['Depth'] = len(block_summary['Layer'])*[i]
            self.module_summary.append(block_summary)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            add_block = UNetUpBlock(prev_shape, 2**(wf+i), up_mode,
                                            padding, batch_norm, kernel_size=kernel_size)
            self.up_path.append(add_block)
            block_summary = add_block.block_summary
            block_summary['Depth'] = len(block_summary['Layer'])*[i]
            self.module_summary.append(block_summary)
            prev_shape = add_block.outp_shape

        self.last = nn.Conv2d(prev_shape[0], n_classes, kernel_size=1)
        out_shape = conv2d_out_shape(prev_shape, n_classes, kernel_size=1, stride=1)
        output_summary = {'Layer':['Output: ' + nn.Conv2d.__name__],'Input shape':[str(prev_shape)],'Output shape':[str(out_shape)], 'Trainable params':[count_module_train_params(self.last)]}
        output_summary['Depth'] = [0]
        self.module_summary.append(output_summary)
        
        if name is not None:
            assert isinstance(name, str)
            self.name_ = name
        
    def summary(self):
        """
        Print a Keras-like summary of the model
        """
        w = 20
        total_params = 0
        # Very dirty stuff
        print(" ")
        print("U-Net '{}'".format(self.name()))
        # Creating the header line
        header = ['Depth', 'Layer','Input shape','Output shape', 'Trainable params']
        
        head_line = ""
        for h in header:
            head_line += h + " "*(w-len(h))
        # Print the header line
        print(head_line)
        for d in self.module_summary:
            di = d[header[0]]
            li = d[header[1]]
            out_si = d[header[3]]
            in_si = d[header[2]]
            t_params = d[header[4]]

            for depth, l, out_s, in_s,p in zip(di,li, out_si, in_si, t_params):
                print(('{0}'+" "*(w-len(str(depth)))+'{1}'+" "*(w-len(l))+'{2}'+" "*(w-len(in_s))+'{3}'+" "*(w-len(out_s))+ '{4}').format(depth,l, in_s, out_s, p))
                total_params += p
            print("")   
        print("-"*len(header)*w)
        print("Total number of trainable parameters: {0}".format(total_params))
        print("-"*len(header)*w)
        
    def name(self):
        if hasattr(self, 'name_'):
            return self.name_
        else:
            return 'Unnamed'
                
    def set_nbr_trainable_params_(self):
        """
            Sets the attrubute num_parameters (total number of trainable parameters
            of the model)
        """
        total_params = 0
        for d in self.module_summary:
            params = d['Trainable params']
            for p in params:
                total_params += p
        self.num_parameters = total_params

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_shape, out_size, padding, batch_norm, kernel_size=3):
        super(UNetConvBlock, self).__init__()
        block = []
        
        # First conv layer
        block.append(nn.Conv2d(in_shape[0], out_size, kernel_size=kernel_size,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        # Calculate output size
        outp_shape = conv2d_out_shape(in_shape, out_size, kernel_size=kernel_size, 
                                      padding=padding, conv_type='down')
        
        
        block.append(nn.Conv2d(out_size, out_size, kernel_size=kernel_size,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        
        outp_shape = conv2d_out_shape(outp_shape, out_size, kernel_size=kernel_size, 
                                      padding=padding, conv_type='down')
        self.outp_shape = outp_shape
        self.block = nn.Sequential(*block)
        
        block_summary = {'Depth':[], 'Layer':[],'Input shape':[],'Output shape':[], 'Trainable params':[]}
        add_to_summary(block_summary, 'ConvBlock', in_shape, outp_shape, self.trainable_params_())
        self.block_summary = block_summary
        
    def trainable_params_(self):
        return count_module_train_params(self)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_shape, out_size, up_mode, padding, batch_norm, kernel_size=3):
        super(UNetUpBlock, self).__init__()
        
        block_summary = {'Depth':[], 'Layer':[],'Input shape':[],'Output shape':[], 'Trainable params':[]}
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_shape[0], out_size, kernel_size=2,
                                         stride=2)
            upconv_outp_shape = conv2d_out_shape(in_shape, out_size, kernel_size=2, stride=2,
                                      padding=0, conv_type='up')
            upconv_nparam = self.trainable_params_()
            add_to_summary(block_summary, 'ConvTransposeLayer', in_shape, upconv_outp_shape, upconv_nparam)
            
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_shape[0], out_size, kernel_size=1))
        
        self.conv_block = UNetConvBlock(in_shape, out_size, padding, batch_norm, kernel_size=kernel_size)
        outp_shape = conv2d_out_shape(upconv_outp_shape, out_size, kernel_size=kernel_size, 
                                      padding=padding, conv_type='down')
        
        add_to_summary(block_summary, 'ConvLayer', upconv_outp_shape, outp_shape, self.trainable_params_()-upconv_nparam)
        self.outp_shape = outp_shape
        self.block_summary = block_summary

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]
    
    def trainable_params_(self):
        return count_module_train_params(self)

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out