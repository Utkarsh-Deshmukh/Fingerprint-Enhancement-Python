# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 03:15:03 2016

@author: utkarsh
"""


# RIDGEFILTER - enhances fingerprint image via oriented filters
#
# Function to enhance fingerprint image via oriented filters
#
# Usage:
#  newim =  ridgefilter(im, orientim, freqim, kx, ky, showfilter)
#
# Arguments:
#         im       - Image to be processed.
#         orientim - Ridge orientation image, obtained from RIDGEORIENT.
#         freqim   - Ridge frequency image, obtained from RIDGEFREQ.
#         kx, ky   - Scale factors specifying the filter sigma relative
#                    to the wavelength of the filter.  This is done so
#                    that the shapes of the filters are invariant to the
#                    scale.  kx controls the sigma in the x direction
#                    which is along the filter, and hence controls the
#                    bandwidth of the filter.  ky controls the sigma
#                    across the filter and hence controls the
#                    orientational selectivity of the filter. A value of
#                    0.5 for both kx and ky is a good starting point.
#         showfilter - An optional flag 0/1.  When set an image of the
#                      largest scale filter is displayed for inspection.
# 
# Returns:
#         newim    - The enhanced image
#
# See also: RIDGEORIENT, RIDGEFREQ, RIDGESEGMENT

# Reference: 
# Hong, L., Wan, Y., and Jain, A. K. Fingerprint image enhancement:
# Algorithm and performance evaluation. IEEE Transactions on Pattern
# Analysis and Machine Intelligence 20, 8 (1998), 777 789.

### REFERENCES

# Peter Kovesi  
# School of Computer Science & Software Engineering
# The University of Western Australia
# pk at csse uwa edu au
# http://www.csse.uwa.edu.au/~pk



import numpy as np
import scipy;
def ridge_filter(im, orient, freq, kx, ky):
    angleInc = 3;
    im = np.double(im);
    rows,cols = im.shape;
    newim = np.zeros((rows,cols));
    
    freq_1d = np.reshape(freq,(1,rows*cols));
    ind = np.where(freq_1d>0);
    
    ind = np.array(ind);
    ind = ind[1,:];    
    
    # Round the array of frequencies to the nearest 0.01 to reduce the
    # number of distinct frequencies we have to deal with.    
    
    non_zero_elems_in_freq = freq_1d[0][ind]; 
    non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq*100)))/100;
    
    unfreq = np.unique(non_zero_elems_in_freq);

    # Generate filters corresponding to these distinct frequencies and
    # orientations in 'angleInc' increments.
    
    sigmax = 1/unfreq[0]*kx;
    sigmay = 1/unfreq[0]*ky;
    
    sze = np.int(np.round(3*np.max([sigmax,sigmay])));
    
    x,y = np.meshgrid(np.linspace(-sze,sze,(2*sze + 1)),np.linspace(-sze,sze,(2*sze + 1)));
    
    reffilter = np.exp(-(( (np.power(x,2))/(sigmax*sigmax) + (np.power(y,2))/(sigmay*sigmay)))) * np.cos(2*np.pi*unfreq[0]*x); # this is the original gabor filter
    
    filt_rows, filt_cols = reffilter.shape;

    angleRange = np.int(180 / angleInc)

    gabor_filter = np.array(np.zeros((angleRange,filt_rows,filt_cols)));

    for o in range(0, angleRange):
        
        # Generate rotated versions of the filter.  Note orientation
        # image provides orientation *along* the ridges, hence +90
        # degrees, and imrotate requires angles +ve anticlockwise, hence
        # the minus sign.        
        
        rot_filt = scipy.ndimage.rotate(reffilter,-(o*angleInc + 90),reshape = False);
        gabor_filter[o] = rot_filt;
                
    # Find indices of matrix points greater than maxsze from the image
    # boundary
    
    maxsze = int(sze);   

    temp = freq>0;    
    validr,validc = np.where(temp)    
    
    temp1 = validr>maxsze;
    temp2 = validr<rows - maxsze;
    temp3 = validc>maxsze;
    temp4 = validc<cols - maxsze;
    
    final_temp = temp1 & temp2 & temp3 & temp4;    
    
    finalind = np.where(final_temp);
    
    # Convert orientation matrix values from radians to an index value
    # that corresponds to round(degrees/angleInc)    
    
    maxorientindex = np.round(180/angleInc);
    orientindex = np.round(orient/np.pi*180/angleInc);
    
    #do the filtering    
    
    for i in range(0,rows):
        for j in range(0,cols):
            if(orientindex[i][j] < 1):
                orientindex[i][j] = orientindex[i][j] + maxorientindex;
            if(orientindex[i][j] > maxorientindex):
                orientindex[i][j] = orientindex[i][j] - maxorientindex;
    finalind_rows,finalind_cols = np.shape(finalind);
    sze = int(sze);
    for k in range(0,finalind_cols):
        r = validr[finalind[0][k]];
        c = validc[finalind[0][k]];
        
        img_block = im[r-sze:r+sze + 1][:,c-sze:c+sze + 1];
        
        newim[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r][c]) - 1]);
        
    return(newim);    