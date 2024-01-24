# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 12:00:08 2022

@author: Nicolas
"""

import generate_EDSR as gen_EDSR
 
ruta_img = 'images' # path to the masks obtained with VGG annotator (csv)
ruta_masks = 'masks' # Path where the images are located

PATH = '.' 


# Running function

gen_EDSR.SR_EDSR(ruta_masks, ruta_img, PATH)


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries


# Cargar la imagen con PIL
image = Image.open(r"C:\Users\nicol\OneDrive\Escritorio\Monitoring_of_Wound\Super_Resolution_EDSR\\generate_images_sr_img\1024_0.jpg")

# Convertir la imagen en un arreglo NumPy
image_array = np.array(image)


# Crear una figura de Matplotlib
fig, ax = plt.subplots()

# Graficar la imagen sin marcos
ax.imshow(image_array)

# Eliminar los ejes superiores y derechos
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Guardar la imagen con una resolución de 600 DPI
fig.savefig("ruta_de_guardado.png", dpi=600)
# Mostrar la figura sin marcos
plt.show()
plt.close()



from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

import cv2


image = Image.open(r"C:\Users\nicol\OneDrive\Escritorio\Monitoring_of_Wound\Super_Resolution_EDSR\\generate_images_sr_img\1024_0.jpg")
mask = cv2.imread(r"C:\Users\nicol\OneDrive\Escritorio\Monitoring_of_Wound\Super_Resolution_EDSR\\\generate_images_sr_mask\1024_0.jpg", cv2.IMREAD_GRAYSCALE)

# Convertir la imagen en un arreglo NumPy
image_array = np.array(image)
other_mask =  mask/255

# Aplicar el umbral (threshold)
_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

img_overlay = gen_EDSR.overlay_mask(image_array.copy(),other_mask)

# Aplicar SLIC a la imagen
segments_slic = slic(img_overlay, n_segments=100, compactness=5, sigma=1)



#obtengo los contornos
image_slic = mark_boundaries(image_array, segments_slic)



plt.imshow(img_overlay)


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
