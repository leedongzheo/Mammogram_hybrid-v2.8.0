#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized training config
Batch size = 64, Epoch = 300
Stable LR schedule to avoid loss explosion.
"""

import time
import os
import torch
import torch.nn as nn
from torch import optim
from general_dataset import Dataset_SEGCLS_png
from loss import MultiOutputLoss, SparseMILLoss
from trainer import Trainer
import model as M
import factory as fm
import misc


device = torch.device('cuda:0')
timestr = time.strftime('%m%d%H%M')

# ================================
# DATASET
# ================================
dh_kwargs = {
        'root': '../data',
        'train_split': 'split/train.txt',
        'val_split': 'split/val.txt',
        'test_split': 'split/test.txt',
        'datapath': '.',
        'train_batchsize': 64,          # <<< UPDATED
        'test_batchsize': 32,
        'modalities': ('label', 'img'),
        'rand_flip': (1, 1),
        'crop_type': 'balance',
        'balance_rate': 0.5,
        'crop_size_img': (512, 384),
        'mini_positive': 100,
        'crop_type_test': 'center',
        'crop_size_img_test': (512, 384),
        'collate_max_hw': (512, 384),
        'DataSet': Dataset_SEGCLS_png,
        'num_workers': 4,
        }

data_cube = misc.DataHub(**dh_kwargs)

# ================================
# LEARNING RATE (stable)
# ================================
lr = 0.01   # SGD default: stable even with batch = 64

lr_scheme = {
        'base_lr': lr,
        'lr_policy': 'multistep',
        'gamma': 0.3,

        # <<< UPDATED: stepvalue theo epoch (ổn định hơn)
        # 300 epoch → giảm LR ở 100 – 200 – 260
        'stepvalue': (100, 200, 260),

        'max_epoch': 300,     # <<< UPDATED
        }

# ================================
# MODEL
# ================================
mil_downfactor = 128
drop = (.2,)*3 + (.5,)*5 + (.2,)*3
model = M.uresnet_16x11x2(1, 2, drop, mil_downfactor, upsampler=fm.BilinearUp2d())
num_mo = model.N + 1

experiment_id = 'UResHDS_%s' % timestr

model_cube = {
        'model': model,
        'init_func': misc.weights_init,
        'pretrain': None,
        'resume': None,
        'optimizer': optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=5e-4
        ),
        'num_mo': num_mo
        }

# ================================
# LOSS WEIGHT (giữ nguyên – đang tốt)
# ================================
loss_weight = (3.5, 3., 2.5, 2., 1.5, 1.)

# update decay theo max_epoch mới
lw_decay_seg = pow(5e-3, 1/lr_scheme['max_epoch'])
lw_decay_cls = pow(5e-2, 1/lr_scheme['max_epoch'])

weight_seg = torch.tensor((1., 1.)).to(device)
weight_cls = torch.tensor((1., 1.)).to(device)

criterion_cube = {
        'criterion_seg': MultiOutputLoss(
                nn.CrossEntropyLoss(weight_seg),
                loss_weight, device, lw_decay_seg
        ),
        'criterion_cls': MultiOutputLoss(
                SparseMILLoss(1e-6, weight_cls),
                loss_weight, device, lw_decay_cls
        ),
        'segcls_weight': (1.0, 0.03),
        }

# ================================
# SNAPSHOT / DISPLAY
# ================================
snapshot_root = './snapshot/%s' % (experiment_id)
os.makedirs(snapshot_root, exist_ok=True)

snapshot_scheme = {
        'root': snapshot_root,
        'display_interval': 10,   # mỗi 10 batch in log
        'test_interval': 50,      # test mỗi ~50 batch để tránh chậm
        'snapshot_interval': 999999,
        }

# ================================
# TRAIN
# ================================
clsfromseg = (criterion_cube['segcls_weight'][1] == 0.0)

trainer = Trainer(
        model_cube, data_cube, criterion_cube,
        lr_scheme, snapshot_scheme, device,
        clsfromseg=clsfromseg
)

trainer.train()

# ================================
# FINAL EVALUATION
# ================================
is_indiv = False
is_save_png = False
trainer.test('cls_max', 'cls_max', is_indiv, is_save_png)
trainer.test('seg_max', 'seg_max', is_indiv, is_save_png)
