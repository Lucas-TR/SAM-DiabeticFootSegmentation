# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 22:04:16 2022

@author: nicolas
"""

import unet
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from time import time


tiempo_inicial = time() 
#Directorios

#train
ruta_train = r"D:\alberta\train_2_UNET\data_train_globular\images_train_all"
ruta_mask_train = r"D:\alberta\train_2_UNET\data_train_globular\masks_train_all" 
#valid
ruta_val = r"D:\alberta\train_2_UNET\data_train_globular\images_val"
ruta_mask_val = r"D:\alberta\train_2_UNET\data_train_globular\masks_val" 
#ruta_test = "/content/drive/MyDrive/Foot Ulcer Segmentation Challenge/test/images"


#Cargamos las imágenes de train, valid y test al programa en forma de arreglos de numpy
Images_train = unet.encoder_images(ruta_train)
Images_mask_train = unet.encoder_masks(ruta_mask_train)

Images_val = unet.encoder_images(ruta_val)
Images_mask_val = unet.encoder_masks(ruta_mask_val)


#Unimos todas las impagenes en forma de arreglo
imgs = np.concatenate([Images_train,Images_val])
masks= np.concatenate([Images_mask_train,Images_mask_val])


#Covertimos los arreglos a tensores y cambiamos el orden de su forma
imgs = torch.from_numpy(imgs).permute(0,3,1,2)
masks = torch.from_numpy(masks).permute(0,3,1,2)


#Cambiamos el tipo de datos a float
imgs = imgs.float()
masks = masks.float()

#obtenemos la cantidad de imagenes para el test
n_test = len(Images_val)


#Dividimos nuestra data en entrenamiento y test y los colocamos en un diccionario
dataset = {
    'train': unet.Dataset(imgs[:-n_test], masks[:-n_test]),
    'val': unet.Dataset(imgs[-n_test:], masks[-n_test:])
}

#len(dataset['train']), len(dataset['test'])

#Definimos un diccionario para cargar nuestros datos y definimos los batch
dataloader = {
    'train': torch.utils.data.DataLoader(dataset['train'], batch_size=32, shuffle=True, pin_memory=True),
    'val': torch.utils.data.DataLoader(dataset['val'], batch_size=4, pin_memory=True)
}

#cargamos el modelo Umet
model = unet.UNet()

#output = model(torch.randn((10,3,394,394)))

#entrenamos el modelo Unet con 200 epocas
hist = unet.fit_all(model, dataloader, epochs=20)


#mostramos entrenamiento
df = pd.DataFrame(hist)
df.plot(grid=True)
plt.show()
plt.close()

#Guardamos el modelo de entrenamiento con extensión .pt y también las gráficas
torch.save(model, 'model.pt')
np.save('hist.npy', hist)

tiempo_final = time() 

tiempo_ejecucion = tiempo_final - tiempo_inicial
print("Tiempo de ejecucion: {}".format(tiempo_ejecucion))