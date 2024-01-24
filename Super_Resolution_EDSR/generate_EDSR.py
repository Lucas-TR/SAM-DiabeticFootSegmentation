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

def overlay_mask(image, mask, mask_color=(0, 0, 0), alpha=0.1):
    # Crear una imagen RGB de la misma forma que la imagen original
    mask_rgb = np.zeros_like(image)

    # Asegurarse de que la máscara es binaria
    mask_binary = (mask > 0).astype(np.uint8)

    # Colorear los píxeles de interés en la máscara
    mask_rgb[mask_binary == 1] = mask_color

    # Superponer la máscara a la imagen utilizando una mezcla ponderada
    overlay = cv2.addWeighted(image, 1, mask_rgb, alpha, 0)

    return overlay


def SR_EDSR(labels, images, PATH, both = True):
    file_img = 'generate_images_sr_img'
    if (both):
        file_mask = 'generate_images_sr_mask'
        tool.new_file(PATH, file_mask, 0)
        
    tool.new_file(PATH, file_img, 0)
    # specifying the zip file nam    
    masks = tool.encoder('{}/{}'.format(PATH,labels))
    imgs = tool.encoder('{}/{}'.format(PATH,images))
    
    names = tool.names_sort(images)
    names = tool.split_dot(names)
    n_iter = len(imgs)
    
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = '{}/{}'.format(PATH,"EDSR_x4.pb")
    sr.readModel(path)
    sr.setModel("edsr",4)
    path_save_img = '{}/{}'.format(PATH,file_img)
    path_save_mask = '{}/{}'.format(PATH,file_mask)
    
    
    for i in range(n_iter):
        result = sr.upsample(imgs[i])
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        name = names[i]
        tool.save_img(path_save_img, result, name)
        if (both):
            result = sr.upsample(masks[i])
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            name = names[i]
            tool.save_img(path_save_mask, result, name)
    
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