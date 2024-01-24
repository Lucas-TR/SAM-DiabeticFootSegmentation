# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 00:42:40 2023

@author: nicol
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import math

def save_images_filtro_specular(imagenes, mascaras, carpeta_origen,  carpeta_destino, carpeta_destino2):
    lista_archivos = os.listdir(carpeta_origen)
    cont = 0
    for imagen, mascara in tqdm(zip(imagenes, mascaras), desc='Aplicando filtro specular', total=len(imagenes)):
        ruta_archivo = os.path.join(carpeta_origen, lista_archivos[cont])
        # Convertir la máscara en una imagen binaria
        imagen_completada, thresholded_img = filtro_specular(imagen, mascara)
        # Obtener el nombre de archivo y extensión de la imagen original
        nombre_archivo, extension = os.path.splitext(os.path.basename(ruta_archivo))
        # Crear el nombre de archivo y la ruta de la imagen recortada en la carpeta de destino
        nombre_archivo_recortado = nombre_archivo + extension
        ruta_archivo_recortado = os.path.join(carpeta_destino, nombre_archivo_recortado)
        ruta_archivo_recortado2 = os.path.join(carpeta_destino2, nombre_archivo_recortado)
        # Guardar la imagen recortada en la carpeta de destino
        cv2.imwrite(ruta_archivo_recortado, imagen_completada)
        cv2.imwrite(ruta_archivo_recortado2, thresholded_img*255)
        cont += 1

def filtro_specular(img, mask):
    """
    Aplica un filtro para eliminar la reflexión especular de las imágenes.
    :param img: imagen original.
    :param mask: máscara de la región de interés.
    :return: imagen con reflexión especular filtrada y máscara umbralizada.
    """

    # Aplicar umbral a la máscara para obtener una versión binaria
    umbral, mascara_filtrada = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

    # Aplicar la máscara binaria a la imagen original
    img_segment = cv2.bitwise_and(img, img, mask=mascara_filtrada)

    # Convertir la imagen segmentada a diferentes espacios de color
    hsv = cv2.cvtColor(img_segment, cv2.COLOR_BGR2HSV)
    green = cv2.cvtColor(img_segment, cv2.COLOR_BGR2RGB)[:,:,1] / 255.0
    lightness = cv2.cvtColor(img_segment, cv2.COLOR_BGR2LAB)[:,:,0] / 100.0

    # Calcular la media del canal verde y aplicar la función exponencial
    green_mean = np.mean(img[:,:,1][mascara_filtrada > 0])
    exponent = exp_green(green_mean)
    
    # Calcular la imagen de características utilizando la ecuación proporcionada
    feature_img = ((1 - hsv[:,:,1] / 255.0) * green * lightness)**exponent

    # Aplicar un filtro de desviación estándar y normalizar
    filtered_img = cv2.GaussianBlur(feature_img, (3,3), cv2.BORDER_DEFAULT)
    mean, std_dev = cv2.meanStdDev(filtered_img)
    normalized_img = (filtered_img - mean) / std_dev

    # Umbralizar la imagen normalizada
    _, thresholded_img = cv2.threshold(normalized_img, 0.3, 1, cv2.THRESH_BINARY)
    thresholded_img = thresholded_img.astype(np.uint8)

    # Aplicar la técnica de inpainting para completar la imagen
    completed_image = cv2.inpaint(img, thresholded_img, 10, cv2.INPAINT_NS)

    return completed_image, thresholded_img


#relación logarítmica que representa el cambio de escala entre la escala 1 que va desde 8 a 200 y la escala 2 que va desde 3 a 10 es:
def exp_green(x1):
    """
    Calcula el exponente para el ajuste de escala basado en la media del canal verde.
    :param x1: media del canal verde.
    :return: valor del exponente calculado.
    """
    a = (math.log(10) - math.log(3)) / (math.log(200) - math.log(8))
    b = math.log(3) - a * math.log(8)
    return math.exp(a * math.log(x1) + b)

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


def get_mask_not_specular(mask, mask_denoise):
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)/255
    mask_np = np.array(mask_gray)

    # Obtén las posiciones donde la máscara es igual a 1
    y_positions, x_positions = np.where(mask_np == 1)

    for x, y in zip(x_positions, y_positions):
            if mask_denoise[y][x] == 1:
                mask_denoise[y][x] = 0
            else:
                mask_denoise[y][x] = 1

    mask_denoise = mask_denoise*255
    mascara_filtrada = cv2.cvtColor(mask_denoise, cv2.COLOR_GRAY2RGB)
    
    return mascara_filtrada

def save_mask_not_specular(imagenes, mascaras, carpeta_origen, carpeta_destino):
    lista_archivos = os.listdir(carpeta_origen)
    cont = 0
    for imagen, mascara in tqdm(zip(imagenes, mascaras), desc='Aplicando filtro specular', total=len(imagenes)):
        ruta_archivo = os.path.join(carpeta_origen, lista_archivos[cont])
        # Convertir la máscara en una imagen binaria
        _, mask_denoise = filtro_specular(imagen, mascara)
        new_mask = get_mask_not_specular(mascara, mask_denoise)
        # Obtener el nombre de archivo y extensión de la imagen original
        nombre_archivo, extension = os.path.splitext(os.path.basename(ruta_archivo))
        # Crear el nombre de archivo y la ruta de la imagen recortada en la carpeta de destino
        nombre_archivo_recortado = nombre_archivo + extension
        ruta_archivo_recortado = os.path.join(carpeta_destino, nombre_archivo_recortado)
        # Guardar la imagen recortada en la carpeta de destino
        cv2.imwrite(ruta_archivo_recortado, new_mask)
        cont += 1

def mean_green(imagenes, mascaras):
    means_green = []
    for imagen, mascara in tqdm(zip(imagenes, mascaras), desc='Calculando el valor minimo y maximo de la media del color verde', total=len(imagenes)):
        # Aplicar un filtro a la máscara
        umbral, mascara_filtrada = cv2.threshold(mascara, 100, 255, cv2.THRESH_BINARY)
    
        # Convertir la máscara filtrada a escala de grises
        mascara_gris = cv2.cvtColor(mascara_filtrada, cv2.COLOR_BGR2GRAY)
        # Extraer los valores del canal verde en las posiciones de la máscara
        green_pixels = imagen[:,:,1][mascara_gris > 0]
    
        # Calcular la media de los valores del canal verde
        mean_green = np.mean(green_pixels)
        means_green.append(mean_green)
            
    return min(means_green), max(means_green), means_green



