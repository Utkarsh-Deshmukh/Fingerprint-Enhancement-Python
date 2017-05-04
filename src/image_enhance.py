# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 22:50:30 2016

@author: utkarsh
"""
from ridge_segment import ridge_segment
from ridge_orient import ridge_orient
from ridge_freq import ridge_freq
from ridge_filter import ridge_filter

def image_enhance(img):
    blksze = 16;
    thresh = 0.1;
    normim,mask = ridge_segment(img,blksze,thresh);             # normalise the image and find a ROI


    gradientsigma = 1;
    blocksigma = 7;
    orientsmoothsigma = 7;
    orientim = ridge_orient(normim, gradientsigma, blocksigma, orientsmoothsigma);              # find orientation of every pixel


    blksze = 38;
    windsze = 5;
    minWaveLength = 5;
    maxWaveLength = 15;
    freq,medfreq = ridge_freq(normim, mask, orientim, blksze, windsze, minWaveLength,maxWaveLength);    #find the overall frequency of ridges
    
    
    freq = medfreq*mask;
    kx = 0.65;ky = 0.65;
    newim = ridge_filter(normim, orientim, freq, kx, ky);       # create gabor filter and do the actual filtering
    
    
    #th, bin_im = cv2.threshold(np.uint8(newim),0,255,cv2.THRESH_BINARY);
    return(newim < -3)