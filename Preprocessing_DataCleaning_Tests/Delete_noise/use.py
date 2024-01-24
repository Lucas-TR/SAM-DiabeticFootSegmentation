# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 03:08:12 2023

@author: nicol
"""

import detect_noise
import cv2
import matplotlib.pyplot as plt
import numpy as np


carpeta_origen = 'Masks'
carpeta_destino = 'masks_denoise'

mascaras = detect_noise.read_images(carpeta_origen)

detect_noise.save_denoise(mascaras, carpeta_origen, carpeta_destino)
    



#Calcular iou con las mascras originales

'''

img_copy = closed_mask.copy()


# Obtener los contornos de los agujeros negros
contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Rellenar los contornos con el valor 255 (blanco)
for i, cnt in enumerate(contours):
    cv2.drawContours(img_copy, contours, i, 255, -1, hierarchy=hierarchy)

# Dibujar el contorno original en la imagen rellenada
cv2.drawContours(img_copy, [contours[0]], 0, 0, 2)

plt.imshow(img_copy)



# Realiza la binarización de la imagen
ret, thresh = cv2.threshold(closed_mask, 127, 255, cv2.THRESH_BINARY)

# Encuentra los contornos de la imagen binarizada
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Encuentra el contorno más grande
largest_contour = max(contours, key=cv2.contourArea)

# Dibuja el contorno más grande en una imagen en blanco
contour_image = np.zeros_like(closed_mask)
cv2.drawContours(contour_image, [largest_contour], 0, 255, 2)

# Muestra la imagen con el contorno encontrado
plt.imshow(contour_image, cmap='gray')
plt.show()

'''
'''


position = np.where(contour_image == 255)

y, x = position







PRUEBAS


filtered_mask = cv2.medianBlur(closed_mask, 3)
plt.imshow(filtered_mask)




############
plt.imshow(mask_prueba)
# Convertir la imagen a escala de grises
gray = cv2.cvtColor(mask_prueba, cv2.COLOR_BGR2GRAY)

# Aplicar una operación de umbralización
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

# Aplicar una operación de etiquetado de regiones
_, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Definir el umbral de área para eliminar regiones negras de gran tamaño
area_threshold = 500

# Iterar a través de las regiones etiquetadas y eliminar las regiones negras de gran tamaño
for i in range(1, len(stats)):
    if stats[i, cv2.CC_STAT_AREA] > area_threshold:
        binary[labels == i] = 255

# Aplicar una operación de cierre para rellenar cualquier pequeño agujero dentro de las regiones blancas
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Aplicar una operación de apertura para eliminar cualquier pequeña región negra restante
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

# Guardar la nueva máscara blanca como una imagen
plt.imshow(opening)


##################

gray = cv2.cvtColor(mask_prueba, cv2.COLOR_BGR2GRAY)


# Aplica umbralización
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Aplica una operación morfológica de apertura
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Encuentra los contornos de las regiones negras
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define el valor umbral de área para borrar las regiones negras
area_threshold = 200

# Para cada contorno encontrado, calcula su área y borra la región negra si su área es mayor que el umbral predefinido
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > area_threshold:
        cv2.drawContours(mask_prueba, [cnt], 0, (255, 255, 255), -1)


plt.imshow(mask_prueba)
'''