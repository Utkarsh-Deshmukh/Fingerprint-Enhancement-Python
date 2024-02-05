# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:42:58 2016

@author: utkarsh
"""

import sys
import cv2
from fingerprint_image_enhancer import FingerprintImageEnhancer

if __name__ == "__main__":
    image_enhancer = FingerprintImageEnhancer()  # Create object called image_enhancer
    if len(sys.argv) < 2:  # load input image
        print("loading sample image")
        IMG_NAME = "1.jpg"
        img = cv2.imread("images/" + IMG_NAME)
    elif len(sys.argv) >= 2:
        IMG_NAME = sys.argv[1]
        img = cv2.imread("../images/" + IMG_NAME)

    if len(img.shape) > 2:  # convert image into gray if necessary
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    out = image_enhancer.enhance(img)  # run image enhancer
    image_enhancer.save_enhanced_image("enhanced/" + IMG_NAME)  # save output
