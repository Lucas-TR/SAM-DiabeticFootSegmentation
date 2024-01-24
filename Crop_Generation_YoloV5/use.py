# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 22:56:11 2022


This program is for cutouts with labels made by the label Images program for Yolov3, Yolov4, Yolov5 models.

@author: Nicolas
"""
import generator_crops as gen_crop


labels = 'labels_coord' #path where the labels generated with Label Images are located
images = 'labels_mask' #Path where the images are located

#Run function
gen_crop.generator_crops(labels, images)