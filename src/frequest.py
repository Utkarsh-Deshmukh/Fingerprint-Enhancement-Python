# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 02:51:53 2016

@author: utkarsh
"""



# FREQEST - Estimate fingerprint ridge frequency within image block
#
# Function to estimate the fingerprint ridge frequency within a small block
# of a fingerprint image.  This function is used by RIDGEFREQ
#
# Usage:
#  freqim =  freqest(im, orientim, windsze, minWaveLength, maxWaveLength)
#
# Arguments:
#         im       - Image block to be processed.
#         orientim - Ridge orientation image of image block.
#         windsze  - Window length used to identify peaks. This should be
#                    an odd integer, say 3 or 5.
#         minWaveLength,  maxWaveLength - Minimum and maximum ridge
#                     wavelengths, in pixels, considered acceptable.
# 
# Returns:
#         freqim    - An image block the same size as im with all values
#                     set to the estimated ridge spatial frequency.  If a
#                     ridge frequency cannot be found, or cannot be found
#                     within the limits set by min and max Wavlength
#                     freqim is set to zeros.
#
# Suggested parameters for a 500dpi fingerprint image
#   freqim = freqest(im,orientim, 5, 5, 15);
#
# See also:  RIDGEFREQ, RIDGEORIENT, RIDGESEGMENT

### REFERENCES

# Peter Kovesi 
# School of Computer Science & Software Engineering
# The University of Western Australia
# pk at csse uwa edu au
# http://www.csse.uwa.edu.au/~pk


import numpy as np
import math
import scipy.ndimage
#import cv2
def frequest(im,orientim,windsze,minWaveLength,maxWaveLength):
    rows,cols = np.shape(im);
    
    # Find mean orientation within the block. This is done by averaging the
    # sines and cosines of the doubled angles before reconstructing the
    # angle again.  This avoids wraparound problems at the origin.
        
    
    cosorient = np.mean(np.cos(2*orientim));
    sinorient = np.mean(np.sin(2*orientim));    
    orient = math.atan2(sinorient,cosorient)/2;
    
    # Rotate the image block so that the ridges are vertical    
    
    #ROT_mat = cv2.getRotationMatrix2D((cols/2,rows/2),orient/np.pi*180 + 90,1)    
    #rotim = cv2.warpAffine(im,ROT_mat,(cols,rows))
    rotim = scipy.ndimage.rotate(im,orient/np.pi*180 + 90,axes=(1,0),reshape = False,order = 3,mode = 'nearest');

    # Now crop the image so that the rotated image does not contain any
    # invalid regions.  This prevents the projection down the columns
    # from being mucked up.
    
    cropsze = int(np.fix(rows/np.sqrt(2)));
    offset = int(np.fix((rows-cropsze)/2));
    rotim = rotim[offset:offset+cropsze][:,offset:offset+cropsze];
    
    # Sum down the columns to get a projection of the grey values down
    # the ridges.
    
    proj = np.sum(rotim,axis = 0);
    dilation = scipy.ndimage.grey_dilation(proj, windsze,structure=np.ones(windsze));

    temp = np.abs(dilation - proj);
    
    peak_thresh = 2;    
    
    maxpts = (temp<peak_thresh) & (proj > np.mean(proj));
    maxind = np.where(maxpts);
    
    rows_maxind,cols_maxind = np.shape(maxind);
    
    # Determine the spatial frequency of the ridges by divinding the
    # distance between the 1st and last peaks by the (No of peaks-1). If no
    # peaks are detected, or the wavelength is outside the allowed bounds,
    # the frequency image is set to 0    
    
    if(cols_maxind<2):
        freqim = np.zeros(im.shape);
    else:
        NoOfPeaks = cols_maxind;
        waveLength = (maxind[0][cols_maxind-1] - maxind[0][0])/(NoOfPeaks - 1);
        if waveLength>=minWaveLength and waveLength<=maxWaveLength:
            freqim = 1/np.double(waveLength) * np.ones(im.shape);
        else:
            freqim = np.zeros(im.shape);
        
    return(freqim);
    