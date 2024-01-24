# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:43:26 2022

@author: nicol
"""

from skimage.segmentation import  slic
from skimage.segmentation import  mark_boundaries
import matplotlib.pyplot as plt
import tool
import cv2



ruta_images = 'images'
save_file = 'Test_sp'
path_out = '.'


tool.new_file(path_out, save_file, 0)
images = tool.encoder(ruta_images)
arr_names = tool.names_sort(ruta_images)
names = tool.split_dot(arr_names)
cont = 0
n_segment = 100; compactness=10; sigma=1
for img in images:
    segments_slic = slic(img, n_segment , compactness=compactness , sigma = sigma)
    boundaries = mark_boundaries(img,segments_slic)
    boundaries = boundaries*255
    
    file = "{}/{}.jpg".format('{}/{}'.format(path_out, save_file), names[cont])
    cv2.imwrite(file,boundaries[:,:,(2,1,0)])
    cv2.waitKey(0)
    cont = cont + 1
     


