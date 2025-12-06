
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 06 12:13:12 2018

@author: Rongzhao Zhang
"""

from __future__ import print_function
import os
import os.path as P
import numpy as np
import imageio.v2 as imageio
import torch
import torch.utils.data as udata

__all__ = ['Dataset_SEGCLS_png', 'MMDataset_memmap', ]


class General_Dataset_SEGCLS(udata.Dataset):
    """
    General Dataset
    INPUT:
        ROOT       -- Root directory of 'split' and 'datapath'
        SPLIT      -- Relative path of split file
        DATAPATH   -- Relative path of data folder
        MODALITIES -- Tuple of modality names, label must be the first one
        TRANSFORM  -- Transform imposed on input data (default = None)

    OUTPUT:
        IMG  -- Tensor (C, H, W), float32
        LABEL -- Tensor (H, W), long, binary (0/1)
        CLS_LABEL -- 0 or 1 (image-level label)
    """
    def __init__(self, root, split, datapath, modalities, transform=None, shape=None):
        self.transform = transform
        self.data = []
        self.label = []
        
        # load list of sample names
        split_fname = os.path.join(root, split)
        sn_list = open(split_fname, 'r').read().splitlines()
        
        for sn in sn_list:
            # read mask
            label_ = self.access_data(root, datapath, modalities[0], sn, 'uint8', shape)
            self.label.append(label_)
            
            # read image modalities
            img_ = []
            for mod in modalities[1:]:
                image = self.access_data(root, datapath, mod, sn, 'float32', shape)
                img_.append(image)
            img_ = np.stack(img_)  # (C, H, W)
            self.data.append(img_)
            

    def access_data(self, root, datapath, mod, sn, dtype, shape=None):
        # implemented in subclass
        pass
    

    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
    
        # 1) BINARY HOÁ LABEL: pixel > 0 → 1, pixel == 0 → 0
        label = (label > 0).astype(np.uint8)
    
        # 2) Chuyển sang Tensor LONG
        label = torch.from_numpy(label).long()
    
        # 3) Transform
        if self.transform is not None:
            img, label = self.transform(img, label)
        else:
            img = torch.from_numpy(img).float()
    
        # 🔴 RẤT QUAN TRỌNG: đảm bảo contiguous để DataLoader không crash
        img = img.contiguous()
        label = label.contiguous()
    
        # 4) Image-level label (0/1)
        cls_label = 1 if label.sum() > 0 else 0
        cls_label = torch.tensor(cls_label).long()
    
        return img, label, cls_label

    

    def __len__(self):
        return len(self.data)



class Dataset_SEGCLS_png(General_Dataset_SEGCLS):
    def access_data(self, root, datapath, mod, sn, dtype, shape=None):
        fname = P.join(root, datapath, mod, '%s.png' % sn)
        data = imageio.imread(fname)
        return data.astype(dtype)



class MMDataset_memmap(udata.Dataset):
    """
    Multi-Modality Dataset (binary mask + image .dat format)
    """
    def __init__(self, root, split, datapath, modalities, transform=None, shape=None):
        self.transform = transform
        self.data = []
        self.label = []

        split_fname = os.path.join(root, split)
        sn_list = open(split_fname, 'r').read().splitlines()

        for sn in sn_list:
            label_fn = os.path.join(root, datapath, modalities[0], '%s.dat' % sn)
            label_ = np.memmap(label_fn, dtype='uint8', mode='r', shape=shape[1:])

            img_fn = os.path.join(root, datapath, modalities[1], '%s.dat' % sn)
            img_ = np.memmap(img_fn, dtype='float32', mode='r', shape=shape)

            self.data.append(img_)
            self.label.append(label_)

    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]

        # Binarize mask
        label = (label > 0).astype(np.uint8)
        label = torch.from_numpy(label).long()

        if self.transform is not None:
            img, label = self.transform(img, label)
        else:
            img = torch.from_numpy(img).float()
        img = img.contiguous()
        label = label.contiguous()
        return img, label

    def __len__(self):
        return len(self.data)
