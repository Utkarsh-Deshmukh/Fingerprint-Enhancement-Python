# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 03:02:23 2016

@author: utkarsh
"""

import numpy as np
#import math
#import scipy.ndimage
from frequest import frequest

def rifdge_freq(im, mask, orient, blksze, windsze,minWaveLength, maxWaveLength):
    rows,cols = im.shape;
    freq = np.zeros((rows,cols));
    
    for r in range(0,rows-blksze,blksze):
        for c in range(0,cols-blksze,blksze):
            blkim = im[r:r+blksze][:,c:c+blksze];
            blkor = orient[r:r+blksze][:,c:c+blksze];
            
            
            freq[r:r+blksze][:,c:c+blksze] = frequest(blkim,blkor,windsze,minWaveLength,maxWaveLength);
    
    freq = freq*mask;
    freq_1d = np.reshape(freq,(1,rows*cols));
    ind = np.where(freq_1d>0);
    
    ind = np.array(ind);
    ind = ind[1,:];    
    
    non_zero_elems_in_freq = freq_1d[0][ind];    
    
    medianfreq = np.median(non_zero_elems_in_freq);
    
    return(medianfreq)