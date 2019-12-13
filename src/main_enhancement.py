# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:42:58 2016

@author: utkarsh
"""

import numpy as np
import cv2
import sys

from image_enhance import image_enhance

if __name__ == '__main__':

    if(len(sys.argv)<2):
        print('loading sample image');
        img_name = '1.jpg'
        img = cv2.imread('../images/' + img_name)
    elif(len(sys.argv) >= 2):
        img_name = sys.argv[1];
        img = cv2.imread('../images/' + img_name)

    if(len(img.shape)>2):
         img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    rows,cols = np.shape(img);
    aspect_ratio = np.double(rows)/np.double(cols);

    new_rows = 350;             # randomly selected number
    new_cols = new_rows/aspect_ratio;

    img = cv2.resize(img,(np.int(new_cols),np.int(new_rows)));

    enhanced_img = image_enhance(img);

    print('saving the image')
    cv2.imwrite('../enhanced/' + img_name, (255*enhanced_img))


