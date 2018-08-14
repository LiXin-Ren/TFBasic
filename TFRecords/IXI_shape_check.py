# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 15:37:04 2018

@author: zxlation
"""

import os
import numpy as np

inDir = 'dataset/IXI/IXI_train_HR/'

npy_list = os.listdir(inDir)
num_npys = len(npy_list)

for i in range(num_npys):
    if (i + 1) % 100 == 0:
        print("%d npys are checked!" % (i + 1))
    volume = np.load(os.path.join(inDir, npy_list[i]))
    [sH, sW, num_slices] = volume.shape
    
    #if (sH != 256) or (sW != 256):
    print("volume%d: (%d, %d, %d)" % (i + 1, sH, sW, num_slices))
    
    