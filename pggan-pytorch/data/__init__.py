"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
from data.pix2pix_dataset import Pix2pixDataset

def create_dataloader(instance, batchSize, is_shuffle, nThreads, isTrain, label_dir, image_dir):

    instance.initialize(label_dir, image_dir)
    print("dataset [%s] of size %d was created" %
          (type(instance).__name__, len(instance)))
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=batchSize,
        shuffle= is_shuffle,
        num_workers=int(nThreads),
        drop_last=False
    )
    return dataloader
