# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:42:58 2016

@author: utkarsh
"""

import sys
from src.FingerprintImageEnhancer import FingerprintImageEnhancer
import cv2

if __name__ == '__main__':

    image_enhancer = FingerprintImageEnhancer()         # Create object called image_enhancer
    if(len(sys.argv)<2):                                # load input image
        print('loading sample image');
        img_name = '2.jpg'
        img = cv2.imread('../images/' + img_name)
    elif(len(sys.argv) >= 2):
        img_name = sys.argv[1];
        img = cv2.imread('../images/' + img_name)

    if(len(img.shape)>2):                               # convert image into gray if necessary
         img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    out = image_enhancer.enhance(img, resize=False)     # run image enhancer
    image_enhancer.save_enhanced_image('../enhanced/' + img_name)   # save output
