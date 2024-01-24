# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:30:56 2022

@author: Nicolas
"""

import IoU_metric as iou
import cv2

ruta_imgs = 'img'
ruta_labels =  'label'


img = cv2.imread('1.png')
label = cv2.imread('1_ref.png')

# Run fuction for one image
iou_value = iou.calculate_iou(img, label)

# Run fuction for a file
iou_value = iou.iou_calculate(ruta_imgs, ruta_labels)