# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 23:05:07 2022

@author: nicolas
"""

#import tool
import torch
import utils

import cv2 #Open CV Python
import numpy 
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

import torch
import pandas as pd
import tool_unet
from shutil import rmtree
import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
sam_checkpoint = "sam_model_best_200.pth"
model_type = "vit_b"


device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

display = utils.notebook_init()  # checks



# the variable methos can be 1, 2, 3, 4

# 1: RGB
# 2: VGG
# 3: RGB and VGG
# 4: unet
method = 3

confiabilidad = 0.20


#instanciando
path_out = "."
path_img = "generate_images_sr_img"
crops = 'crops'
path_out_origin = path_out
#path yolov5 model
Path = '{}/{}'.format(path_out, 'runs/train/exp/weights/best.pt')

#cargando las imagenes
ruta_images = 'images'
ruta_masks = 'masks'
images = tool.encoder(ruta_images)
masks = tool.encoder(ruta_masks)



names_imgs = os.listdir(ruta_images)
names_imgs.sort()


dic_model = {1: 'RGB', 2: 'VGG', 3: 'RGB_VGG'}
ruta_carpeta = 'exp_final_fine_tunin_points{}'.format(dic_model[method])
tool.create_new_file(ruta_carpeta)
tool.create_new_file(crops)
iou_total = []



for i in tqdm(range(len(images)), desc="Processing images"):
    
    # ruta de la imagen original
    src = 'images/{}'.format(names_imgs[i])
    
    # ruta donde quieres mover la imagen
    dst = 'data/images/{}'.format(names_imgs[i])
    
    # Copiar y pegar las imagenes para que yolo pueda detectarlo
    shutil.copy(src, dst)
    
    masks_pred = []
    
    #Realizamos la detección de lesiones con el modelo Yolo v5
    h, w, _ = images[i].shape
    bboxes = detect.run(weights = Path, conf_thres = confiabilidad, save_crop=False)
    
    data_images = tool_SVM.encoder_files(path_img)
    name_images = tool_SVM.split_dot(data_images)
    
    
    for idx in range(len(bboxes)):
        
        #extraemos los bbox de la prediccion de yolov5
        bbox = bboxes[idx][0]
        bbox = np.asarray(bbox, dtype=np.int32)
        
        
        x1, y1, x2, y2 = bbox
        
        #extraermos el recorte
        recorte = images[i][y1:y2, x1:x2, :]
        
        #guardamos las dimensiones
        h_crop, w_crop, _ = recorte.shape
        
        cv2.imwrite('crops/crop.jpg', recorte[:,:,(2,1,0)])
        
        gen_EDSR.SR_EDSR(path_out, 'crops')
        
        
        if (method == 1 or method == 2 or method == 3):
            n_segment = 100; compactness=10; sigma=1 ; threshold = 180; layer=5; file_per_image = True; file_iou_image = True
            mask_sam_p = tool_SVM.prediction_SVM_predict(path_out_origin, path_out, path_img,n_segment, compactness, sigma, threshold , layer, name_images, method, predictor)
        
        
        
        mascara_complete = np.zeros((h, w), dtype=np.uint8)
        
        mask_sam_p = cv2.resize(mask_sam_p, (w_crop, h_crop))
        
        mascara_complete[y1:y2, x1:x2] = mask_sam_p
        
    
        masks_pred.append(mascara_complete)
        
    if len(masks_pred) > 0:
        mask_predict_final = tool.unir_mascaras(masks_pred)
        mask_predict_final = np.squeeze(mask_predict_final)
        mask_predict_final = mask_predict_final.astype(np.uint8) * 255
        
    mask_predict_final = cv2.cvtColor(mask_predict_final, cv2.COLOR_GRAY2RGB)
    
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



