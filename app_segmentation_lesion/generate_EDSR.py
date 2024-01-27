# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:20:41 2022

@author: Nicolas
"""
import cv2
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import  slic, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import tool


    


def SR_EDSR(PATH, images,sr, labels = ''):
    if labels == '':
        both = False
    else:
        both = True
        
    file_img = 'generate_images_sr_img'
    path_save_img = '{}/{}'.format(PATH,file_img)
    if (both):
        file_mask = 'generate_images_sr_mask'
        tool.new_file(PATH, file_mask, 0)
        path_save_mask = '{}/{}'.format(PATH,file_mask)
        
    tool.new_file(PATH, file_img, 0)
    # specifying the zip file nam
    if both:
        masks = tool.encoder('{}/{}'.format(PATH,labels))
    imgs = tool.encoder('{}/{}'.format(PATH,images))
    
    names = tool.names_sort(images)
    names = tool.split_dot(names)
    if len(imgs.shape) != 1:
        n_iter = 1
    else:
        n_iter = len(imgs)
    

    
    
    if n_iter != 0 : #modificar
        for i in range(n_iter):
            if len(imgs.shape) != 1:
                result = sr.upsample(imgs)
            else:
                result = sr.upsample(imgs[i])
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            name = names[i]
            tool.save_img_EDSR(path_save_img, result, name)
            if (both):
                result = sr.upsample(masks[i])
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                name = names[i]
                tool.save_img_EDSR(path_save_mask, result, name)
        
        if (both):
            masks_v = tool.encoder('{}/{}'.format(PATH,file_mask))
            imgs_v = tool.encoder('{}/{}'.format(PATH,file_img))
        
            for j in range(n_iter):
                try:
                    shape_img = imgs_v.shape
                    shape_mask = masks_v.shape
                    if (shape_img != shape_mask):
                        print('la imagen {} no esta cumpliendo'.format(names[j]))
                except ValueError:
                    print('Opps')
        print('succesfull')
    else:
        print('No se encontraron lesiones')


def SR_EDSR_simplify(img, sr):
    # Asumiendo que 'img' es una imagen cargada con cv2.imread o similar
    result = sr.upsample(img)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return result


'''
path = "EDSR_x4.pb"
 
sr.readModel(path)
 
sr.setModel("edsr",4)
 
result = sr.upsample(img)
 
# Resized image
resized = cv2.resize(img,dsize=None,fx=4,fy=4)
 
plt.figure(figsize=(12,8))
plt.subplot(1,3,1)
# Original image
plt.imshow(img[:,:,::-1])
plt.subplot(1,3,2)
# SR upscaled
plt.imshow(result[:,:,::-1])
plt.subplot(1,3,3)
# OpenCV upscaled
plt.imshow(resized[:,:,::-1])
plt.show()


#imge_scale_mask = result

imge_scale = result
'''