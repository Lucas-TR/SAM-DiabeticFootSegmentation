# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 02:20:11 2023

@author: nicol
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


def denoise(img):
    # Convierte la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicamos thresholding automático con el algoritmo de Otsu. ESto hará que el texto se vea blanco, y los elementos
    # del fondo sean menos prominentes.
    #thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    y, x = np.shape(gray)
    val = [int(y/20),int(x/20)]
    xmin = min(val)
    # definir un kernel de 5x5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (xmin, xmin))

    # aplicar el filtrado de apertura morfológica
    closed_mask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)


    # Aplica un umbral para binarizar la imagen
    _, thresh = cv2.threshold(closed_mask, 127, 255, cv2.THRESH_BINARY)

    # Encuentra los contornos en la imagen binarizada
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Encuentra el contorno más grande por área
    largest_contour = max(contours, key=cv2.contourArea)

    # Crea una máscara del tamaño de la imagen
    mask = np.zeros_like(closed_mask)

    # Dibuja el contorno más grande en la máscara con el valor de 1
    cv2.fillPoly(mask, [largest_contour], 1)
    
    mask2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)*255

    #superposed_img = cv2.addWeighted(gray, 0.5, dist, 0.5, 0)
    
    return mask2

def read_images(ruta_directorio):
    """
    Lee un directorio de imágenes en diferentes formatos con OpenCV y devuelve una lista de las imágenes.
    :param ruta_directorio: la ruta del directorio de imágenes.
    :return: una lista de las imágenes.
    """
    imagenes = []
    try:
        # Obtenemos la lista de archivos en el directorio
        lista_archivos = os.listdir(ruta_directorio)
        # Usamos tqdm para mostrar la barra de progreso
        with tqdm(total=len(lista_archivos), desc="Cargando imágenes") as pbar:
            for nombre_archivo in lista_archivos:
                ruta_archivo = os.path.join(ruta_directorio, nombre_archivo)
                imagen = cv2.imread(ruta_archivo)
                if imagen is not None:
                    imagenes.append(imagen)
                pbar.update(1)
    except:
        print("Error al leer el directorio de imágenes.")
    return imagenes


def save_denoise(mascaras, carpeta_origen, carpeta_destino):
    lista_archivos = os.listdir(carpeta_origen)
    cont = 0
    for mascara in tqdm(mascaras, desc='Aplicando filtro denoise', total=len(mascaras)):
        ruta_archivo = os.path.join(carpeta_origen, lista_archivos[cont])
        # Convertir la máscara en una imagen binaria
        closed_mask = denoise(mascara)
        
        # Obtener el nombre de archivo y extensión de la imagen original
        nombre_archivo, extension = os.path.splitext(os.path.basename(ruta_archivo))
        # Crear el nombre de archivo y la ruta de la imagen recortada en la carpeta de destino
        nombre_archivo_recortado = nombre_archivo + extension
        ruta_archivo_recortado = os.path.join(carpeta_destino, nombre_archivo_recortado)
        # Guardar la imagen recortada en la carpeta de destino
        cv2.imwrite(ruta_archivo_recortado, closed_mask)
        cont += 1
     
def grub_cut(img, mask):


    newmask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) #pasar a escala de gris para que solo sea una matriz, pero no esta normalizado


    ret, orig_mask = cv2.threshold(newmask, 20, 255, cv2.THRESH_BINARY) #normalizanos, es decir los pixeles solo tomaran 0 o 255
    orig_mask = orig_mask/255 #los pixeles estaran entre 0 y 1
    
    orig_mask = np.array(orig_mask, dtype=np.uint8)
    
    orig_mask_new = np.zeros(img.shape[:2],np.uint8)
    # donde sea que esté marcado en blanco (primer plano seguro), cambiar mask=1
    # donde sea que esté marcado en negro (fondo seguro), cambiar mask=0
    orig_mask_new[orig_mask == 0] = 2
    orig_mask_new[orig_mask == 1] = 3

    #esto es fijo
    bgdModel = np.zeros((1,65),dtype = np.float64)
    fgdModel = np.zeros((1,65),dtype = np.float64)

    
    cv2.grabCut(img, orig_mask_new , None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)


    mask_grab = np.where((orig_mask_new==2)|(orig_mask_new==0),0,1).astype('uint8')
    
    mask_grab = cv2.cvtColor(mask_grab, cv2.COLOR_GRAY2BGR)
    
    mask_grab =  mask_grab*255
    
    return mask_grab