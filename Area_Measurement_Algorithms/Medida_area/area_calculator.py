# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 17:54:10 2023

@author: nicol
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np

ruta = 'download.png'

img = cv2.imread(ruta)

plt.imshow(img[:,:,(2,1,0)])


img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
verdeBajo1 = np.array([40, 100, 100], np.uint8)
verdeAlto1 = np.array([80, 255, 255], np.uint8)
# Pasamos las imágenes de BGR a: GRAY (esta a BGR nuevamente) y a HSV
imageHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Detectamos el color verde
mask = cv2.inRange(imageHSV, verdeBajo1, verdeAlto1)
area = len(mask[np.where(mask!=0)])
factor = 4/area # en cm^2


plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(mask,cmap="gray")
plt.subplot(1,2,2)
greenDetected = cv2.bitwise_and(img,img,mask=mask)
plt.imshow(greenDetected)
plt.show()
