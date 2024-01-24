# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 23:05:07 2022

@author: nicolas
"""

#import tool
import torch
import utils
import cv2 #Open CV Python
import numpy as np
import os
import matplotlib.pyplot as plt
import detect 
from os import remove
import shutil
from shutil import rmtree
import generate_EDSR as gen_EDSR
import tool_SVM
import tool
import unet
import pandas as pd
import tool_unet
from shutil import rmtree
import sys

# Añadir la ruta del directorio padre al path del sistema
sys.path.append("..")

# Importar el modelo SAM y el predictor de SAM
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
#sam_checkpoint = "sam_vit_h_4b8939.pth"
#model_type = "vit_h"
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

# Definir el dispositivo para la ejecución (GPU)
device = "cuda"

# Inicializar el modelo SAM y moverlo al dispositivo especificado
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
# Crear un predictor SAM
predictor = SamPredictor(sam)

# Inicialización para notebook, útil para visualizaciones
display = utils.notebook_init()  # checks

# Definición de los métodos de segmentación (1: RGB, 2: VGG, 3: RGB y VGG)
method = 3
# Nombre del experimento
name_experiment = 'experiment_rgb_vgg_sam_l'
# Umbral de confianza para la detección con YOLOv5
confiabilidad = 0.20

# Rutas para salida y entrada de imágenes
path_out_origin = "."
path_img = "generate_images_sr_img"
crops = 'crops'

# Ruta al modelo YOLOv5
Path = '{}/{}'.format(path_out_origin, 'runs/train/exp/weights/best.pt')

# Cargar las imágenes
ruta_images = 'images'
ruta_masks = 'masks'
images = tool.encoder(ruta_images)
masks = tool.encoder(ruta_masks)


# Leer los nombres de las imágenes en el directorio
names_imgs = os.listdir(ruta_images)
names_imgs.sort()

# Diccionario para mapear el método de segmentación
dic_model = {1: 'RGB', 2: 'VGG', 3: 'RGB_VGG'}


# Crear rutas y directorios para almacenar resultados
ruta_carpeta = 'predicts_{}_{}'.format(dic_model[method], name_experiment)
tool.create_new_file(ruta_carpeta)
tool.create_new_file(crops)
iou_total = []
path_out = ruta_carpeta
# Creación de varios subdirectorios para diferentes tipos de salidas
tool.create_new_file('{}/{}'.format(ruta_carpeta,'overlay'))
tool.create_new_file('{}/{}'.format(ruta_carpeta,'crops_filter'))
tool.create_new_file('{}/{}'.format(ruta_carpeta,'marcas_crop'))
tool.create_new_file('{}/{}'.format(ruta_carpeta,'marcas_images'))
tool.create_new_file('{}/{}'.format(ruta_carpeta,'slic_boundaries'))
tool.create_new_file('{}/{}'.format(ruta_carpeta,'file_images_centroide_sample'))
tool.create_new_file('{}/{}'.format(ruta_carpeta,'images_bbox'))


for i in tqdm(range(len(images)), desc="Processing images"):
    
    # Definir las rutas de origen y destino para cada imagen.
    # 'src' especifica la ruta de la imagen original.
    # 'dst' es la ruta donde se moverá la imagen para su procesamiento con YOLOv5.
    src = 'images/{}'.format(names_imgs[i])
    dst = 'data/images/{}'.format(names_imgs[i])
    
    # Copiar la imagen desde su ubicación original al destino.
    # Esto es necesario para la compatibilidad con la estructura de directorios de YOLOv5.
    shutil.copy(src, dst)
    
    # Inicializar la lista para almacenar las máscaras de predicción.
    masks_pred = []
    
    # Utilizar YOLOv5 para detectar objetos en la imagen.
    # 'h' y 'w' son las dimensiones de la imagen, utilizadas más adelante para ajustes de escala.
    h, w, _ = images[i].shape
    bboxes = detect.run(weights = Path, conf_thres = confiabilidad, save_crop=False)
    
    data_images = tool_SVM.encoder_files(path_img)
    name_images = tool_SVM.split_dot(data_images)
    
    # Procesar cada bounding box detectado por YOLOv5.
    for idx in range(len(bboxes)):
        
        # Extraer el bounding box y convertirlo a un array de NumPy de tipo entero.
        bbox = bboxes[idx][0]
        bbox = np.asarray(bbox, dtype=np.int32)
        
        # Descomponer el bounding box en sus componentes (x1, y1, x2, y2).
        x1, y1, x2, y2 = bbox
        
        # Extraer el recorte de la imagen basado en el bounding box.
        recorte = images[i][y1:y2, x1:x2, :]
        
        #guardamos las dimensiones
        h_crop, w_crop, _ = recorte.shape
        
        # Guardar el recorte en formato JPEG.
        cv2.imwrite('crops/crop.jpg', recorte[:,:,(2,1,0)])
        
        # Aplicar el algoritmo de super-resolución al recorte y guardar el resultado.
        gen_EDSR.SR_EDSR(path_out_origin, 'crops')
        
        #img_with SR
        recorte_sr = cv2.imread('generate_images_sr_img/crop.jpg')
        
        # Copiar la imagen con super-resolución al directorio correspondiente.
        shutil.copy('generate_images_sr_img/crop.jpg', '{}/{}/{}'.format(ruta_carpeta,'crops_filter', names_imgs[i] ))
        
        
        #guardamos lasss dimensiones
        h_crop_sr, w_crop_sr, _ = recorte_sr.shape 
        
        # Calcular los factores de escala para la imagen con super-resolución.
        fh = h_crop / h_crop_sr
        fw = w_crop/ w_crop_sr
        
        # Realizar la segmentación y predicción utilizando un modelo RL.
        if (method == 1 or method == 2 or method == 3):
            # Configuración de parámetros para la segmentación.
            n_segment = 100; compactness=10; sigma=1 ; threshold = 180; layer=5; file_per_image = True; file_iou_image = True
            input_point_all, input_label_all = tool_SVM.prediction_SVM_predict(path_out_origin, path_out, path_img,n_segment, compactness, sigma, threshold , layer, name_images, method, predictor, names_imgs[i])
        
       
        # Visualizar y guardar la imagen con los puntos de segmentación.
        img_view = tool.show_points_on_image(recorte_sr.copy(), input_point_all, input_label_all)
        cv2.imwrite('{}/{}'.format('{}/{}'.format(ruta_carpeta,'marcas_crop'), names_imgs[i]), img_view)

        # Ajustar las coordenadas de los puntos de segmentación a las dimensiones originales de la imagen.
        for coord in input_point_all:
            coord[0] = x1 + coord[0]*fw
            coord[1] = y1 + coord[1]*fh
         
        # Visualizar y guardar la imagen original con los puntos de segmentación.
        img_view_2 = tool.show_points_on_image(images[i].copy(), input_point_all, input_label_all, complete = True)
        cv2.imwrite('{}/{}'.format( '{}/{}'.format(ruta_carpeta,'marcas_images'),  names_imgs[i]), img_view_2[:,:,(2,1,0)])
        
        # Generar y almacenar la máscara de segmentación.
        mask_pred = tool.segment_image(images[i], input_point_all, input_label_all,  bbox, predictor)
        masks_pred.append(mask_pred)
        
        # Dibujar el bounding box en la imagen y guardar el resultado.
        img_bbox = tool.draw_bbox(img_view_2.copy(), bbox, color=(255, 0, 0))
        cv2.imwrite('{}/{}'.format('{}/{}'.format(ruta_carpeta,'images_bbox'), names_imgs[i]), img_bbox[:,:,(2,1,0)])
        
        
    # Combinar todas las máscaras de segmentación en una sola.
    if len(masks_pred) > 0:
        mask_predict_final = tool.unir_mascaras(masks_pred)
        mask_predict_final = np.squeeze(mask_predict_final)
        mask_predict_final_gray = mask_predict_final.astype(np.uint8) * 255
        
    mask_predict_final = cv2.cvtColor(mask_predict_final_gray, cv2.COLOR_GRAY2RGB)
    
    
    img_overlay = tool.overlay_mask(images[i], mask_predict_final_gray)
    

    # Superponer la máscara final en la imagen original y guardar el resultado.
    cv2.imwrite('{}/{}/{}'.format(ruta_carpeta, 'overlay',names_imgs[i]), img_overlay[:,:,(2,1,0)])
    cv2.imwrite('{}/{}'.format(ruta_carpeta,names_imgs[i]), mask_predict_final)
    iou = tool.calculate_iou(mask_predict_final, masks[i])
    iou_total.append(iou)
    # Borrar el archivo
    os.remove(dst)
    os.remove('crops/crop.jpg')
    

iou_total = np.array(iou_total)
mean = np.mean(iou_total)



# Convertir la lista y el array a Series de pandas:
serie_lista = pd.Series(names_imgs)
serie_array = pd.Series(iou_total)

# Unir las Series en un dataframe:
df = pd.DataFrame({'Names_img': serie_lista, 'IoU': serie_array})


# Guardar el dataframe en un archivo Excel:
df.to_excel('{}/IoU_method_{}.xlsx'.format(ruta_carpeta,dic_model[method]), index=False)


# Abrir el archivo en modo escritura
archivo = open("{}/README.txt".format(ruta_carpeta), "w")

# Escribir "Hola mundo" en el archivo
archivo.write("IoU promedio : {:.3f}".format(mean))



