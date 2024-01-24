# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 01:13:33 2023

@author: nicol
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


# read as grayscale



I = cv2.imread("0012_0.jpg")


def pre_procesing(I):
    
    #formato RGB
    I = I[:,:,(2,1,0)]
    
    bins_edges_min_max = [0, 256]
    num_bins = 256
    bin_count, bins_edges = np.histogram(I, num_bins, bins_edges_min_max)
    
    pdf = bin_count / np.sum(bin_count)
    cdf = np.cumsum(pdf)
    f_eq = np.round(cdf * 300).astype(int)
    
    I_eq = f_eq[I]
    
    
    return I_eq


plt.imshow(I)
image = pre_procesing(I)
plt.imshow(image)

"""

# Creamos una copia para poderla manipular a nuestro antojo.
image_copy = np.copy(I_eq)

# Mostramos la imagen y esperamos que el usuario presione cualquier tecla para continuar.
#cv2_imshow(image)
#cv2.waitKey(0)

# Convertiremos la imagen en un arreglo de ternas, las cuales representan el valor de cada pixel. En pocas palabras,
# estamos aplanando la imagen, volviéndola un vector de puntos en un espacio 3D.
pixel_values = image_copy.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Abajo estamos aplicando K-Means.

# Definimos el criterio de terminación del algoritmo. En este caso, terminaremos cuando la última actualización de los
# centroides sea menor a *epsilon* (cv2.TERM_CRITERIA_EPS), donde epsilon es 1.0 (último elemento de la tupla), o bien
# cuando se hayan completado 10 iteraciones (segundo elemento de la tupla, criterio cv2.TERM_CRITERIA_MAX_ITER).
stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Este es el número de veces que se correrá K-Means con diferentes inicializaciones. La función retornará los mejores
# resultados.
number_of_attempts = 50

# Esta es la estrategia para inicializar los centroides. En este caso, optamos por inicialización aleatoria.
centroid_initialization_strategy = cv2.KMEANS_RANDOM_CENTERS

# Ejecutamos K-Means con los siguientes parámetros:
# - El arreglo de pixeles.
# - K o el número de clusters a hallar.
# - None indicando que no pasaremos un arreglo opcional de las mejores etiquetas.
# - Condición de parada.
# - Número de ejecuciones.
# - Estrategia de inicialización.
#
# El algoritmo retorna las siguientes salidas:
# - Un arreglo con la distancia de cada punto a su centroide. Aquí lo ignoramos.
# - Arreglo de etiquetas.
# - Arreglo de centroides.
_, labels, centers = cv2.kmeans(pixel_values,
                                10,
                                None,
                                stop_criteria,
                                number_of_attempts,
                                centroid_initialization_strategy)

# Aplicamos las etiquetas a los centroides para segmentar los pixeles en su grupo correspondiente.
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# Debemos reestructurar el arreglo de datos segmentados con las dimensiones de la imagen original.
segmented_image = segmented_data.reshape(image_copy.shape)

# Mostramos la imagen segmentada resultante.
plt.imshow(segmented_image)
"""



rojoBajo1 = np.array([0, 100, 20], np.uint8)
rojoAlto1 = np.array([20, 255, 255], np.uint8)

rojoBajo2 = np.array([110, 100, 20], np.uint8)
rojoAlto2 = np.array([255, 255, 255], np.uint8)

#image = segmented_image
image = np.uint8(I_eq)
# Pasamos las imágenes de BGR a: GRAY (esta a BGR nuevamente) y a HSV
#imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#imageGray = cv2.cvtColor(imageGray, cv2.COLOR_GRAY2BGR)
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

maskRojo1 = cv2.inRange(imageHSV, rojoBajo1, rojoAlto1)
maskRojo2 = cv2.inRange(imageHSV, rojoBajo2, rojoAlto2)
mask = cv2.add(maskRojo1,maskRojo2)
mask = cv2.medianBlur(mask, 7)

plt.imshow(mask)
redDetected = cv2.bitwise_and(image,image,mask=mask)

plt.imshow(redDetected)