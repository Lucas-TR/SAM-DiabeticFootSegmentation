# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 01:24:47 2023

@author: nicol
"""

import delete_specular
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

ruta_images = 'Images'
ruta_masks = 'Masks'
carpeta_destino =  'New_Images'
carpeta_destino2 = 'New_Masks'

Images = delete_specular.read_images(ruta_images)
Masks = delete_specular.read_images(ruta_masks)


#Aplicando filtro
delete_specular.save_images_filtro_specular(Images, Masks, ruta_images, carpeta_destino,carpeta_destino2)

#obteniendo mascaras
delete_specular.save_mask_not_specular(Images, Masks, ruta_images, carpeta_destino)

