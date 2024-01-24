# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 02:46:56 2023

@author: nicol
"""

import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Cargar la imagen
image = cv2.imread('0161_0.jpg')

# Aplicar SLIC y obtener los superpixels
segments = slic(img_as_float(image), n_segments = 100, sigma = 1,compactness=5)

# Extraer características (media, varianza y intensidad media del color rojo) para cada superpixel
# y calcular los centroides de los superpíxeles
features = []
centroids = []
for (i, segVal) in enumerate(np.unique(segments)):
    mask = np.zeros(image.shape[:2], dtype = "uint8")
    mask[segments == segVal] = 255

    # Features de color
    mean = cv2.mean(image, mask = mask)[:3]
    var = cv2.meanStdDev(image, mask = mask)[1][:3]
    red_mean = mean[2]
    green_mean = mean[1]
    blue_mean = mean[0]

    # Tamaño del superpíxel (puede ser útil normalizar esto por el tamaño de la imagen)
    size = np.sum(mask) / (image.shape[0] * image.shape[1])

    # Calcular la textura usando algún descriptor, por ejemplo LBP (deberías implementar o importar una función lbp)
    # grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # lbp = lbp(grayscale, mask)

    features.append(np.concatenate([mean, var.flatten(), np.array([red_mean, green_mean, blue_mean, size])], axis=0))  # Añade aquí más características si lo consideras necesario

    # Calcular los centroides
    y_coor, x_coor = np.where(segments == segVal)
    centroids.append((np.mean(x_coor), np.mean(y_coor)))

# Escalar las características
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Aplicar K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(features)

# Usar las etiquetas de K-Means para entrenar la regresión logística
X_train = features
y_train = kmeans.labels_

# Aplicar regresión logística
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# Obtener las probabilidades
probabilities = lr.predict_proba(X_train)

# Encontrar los 5 superpíxeles con mayor probabilidad de ser una lesión
top_5_lesion = np.argsort(probabilities[:, 1])[-5:]

# Encontrar los 5 superpíxeles con mayor probabilidad de NO ser una lesión
top_5_no_lesion = np.argsort(probabilities[:, 0])[-5:]

# Aquí 'centroids' es una lista de los centroides de tus superpíxeles
top_5_lesion_centroids = [centroids[i] for i in top_5_lesion]
top_5_no_lesion_centroids = [centroids[i] for i in top_5_no_lesion]

# Pasar estos centroides a tu modelo SAM...

# Dibujar 'x' azules en los centroides de los superpíxeles seleccionados de la clase "lesión"
# y 'x' rojas en los centroides de los superpíxeles seleccionados de la clase "no lesión"
from skimage.segmentation import mark_boundaries

# Dibujar 'x' azules en los centroides de los superpíxeles seleccionados de la clase "lesión"
# y 'x' rojas en los centroides de los superpíxeles seleccionados de la clase "no lesión"
fig, ax = plt.subplots()

# Mostrar la imagen con los límites de los superpíxeles
marked_image = mark_boundaries(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), segments)
ax.imshow(marked_image)

# Dibujar los centroides de los superpíxeles
# Nota: las coordenadas están invertidas al trazar: (x, y) -> (y, x)
for (x, y) in top_5_lesion_centroids:
    ax.plot(x, y, 'bx')  # x es horizontal (columnas), y es vertical (filas)
for (x, y) in top_5_no_lesion_centroids:
    ax.plot(x, y, 'rx')  # x es horizontal (columnas), y es vertical (filas)

# Quitar los ejes y el espacio en blanco adicional
plt.axis('off')
plt.tight_layout()
plt.show()