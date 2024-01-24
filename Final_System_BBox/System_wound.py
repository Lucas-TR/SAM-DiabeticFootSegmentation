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
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

display = utils.notebook_init()  # checks

#test
path_out = "."

# the variable methos can be 1, 2, 3, 4

# 1: RGB
# 2: VGG
# 3: RGB and VGG
# 4: unet
method = 2

confiabilidad = 0.20


Path = '{}/{}'.format(path_out, 'runs/train/exp/weights/best.pt')

#Cargamos el modelo Unet
ruta_model = '{}'.format('model_3')
model = torch.load('{}/{}'.format(ruta_model,'model.pt'))


ruta_images = 'images'
ruta_masks = 'masks'

images = tool.encoder(ruta_images)
masks = tool.encoder(ruta_masks)


names_imgs = os.listdir(ruta_images)
names_imgs.sort()


ruta_carpeta = 'predicts'
tool.create_new_file(ruta_carpeta)
iou_total = []
tool.create_new_file('{}/{}'.format(ruta_carpeta,'overlay'))
tool.with_mark

for i in tqdm(range(len(images)), desc="Processing images"):
    # ruta de la imagen original
    src = 'images/{}'.format(names_imgs[i])
    
    # ruta donde quieres mover la imagen
    dst = 'data/images/{}'.format(names_imgs[i])
    
    # Copiar la imagen
    shutil.copy(src, dst)
    
    masks_pred = []
    
    #Realizamos la detección de lesiones con el modelo Yolo v5
    bboxes = detect.run(weights = Path, conf_thres = confiabilidad, save_crop=False)
    
    for idx in range(len(bboxes)):
        
        bbox = bboxes[idx][0]
        bbox = np.asarray(bbox, dtype=np.int32)
        
        mask_pred = tool.segment_image(images[i], bbox, predictor)
        masks_pred.append(mask_pred)
        
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

iou_total = np.array(iou_total)
mean = np.mean(iou_total)

# Convertir la lista y el array a Series de pandas:
serie_lista = pd.Series(names_imgs)
serie_array = pd.Series(iou_total)

# Unir las Series en un dataframe:
df = pd.DataFrame({'Names_img': serie_lista, 'IoU': serie_array})


# Guardar el dataframe en un archivo Excel:
df.to_excel('{}/mi_archivo.xlsx'.format(ruta_carpeta), index=False)



'''



ruta_pre_yolo = '{}/{}'.format(path_out,'runs/detect')
detects = os.listdir(ruta_pre_yolo)
file_predict = max(detects)
new_ruta_pre_yolo = '{}/{}/{}/{}'.format('runs/detect', file_predict, 'crops', 'wound')




# Aplicamos la super resolución a la predicción del modelo de Yolo v5
gen_EDSR.SR_EDSR(path_out, new_ruta_pre_yolo)

#borramos los archivos del experimento anterior
rmtree(ruta_pre_yolo)

#almacenando y cargando el modelo
path_img = "generate_images_sr_img"
path_out_origin = path_out
data_images = tool_SVM.encoder_files(path_img)
name_images = tool_SVM.split_dot(data_images)


# segmentation with superpixeles
if (method == 1 or method == 2 or method == 3):
    n_segment = 140; compactness=10; sigma=1 ; threshold = 180; layer=1; file_per_image = True; file_iou_image = True
    tool_SVM.prediction_SVM_predict(path_out_origin, path_out, path_img,n_segment, compactness, sigma, threshold , layer, name_images, method)

# Segmentación con el moddelo Unet
if method == 4:
    Images_test_tensor = unet.encoder_images(path_img)
    Images_test = tool_unet.encoder(path_img)
    
    names_dot = tool_unet.encoder_files(path_img)
    names = tool_unet.split_dot(names_dot)
    
    file = 'Vista'
    tool_unet.new_file(path_out, file, 0)
    new_path_out = '{}/{}'.format(path_out,file)
    
    if len(Images_test.shape) == 3:
       n_iter = 1
    else:
       n_iter = len(Images_test)
    
    for i in range(n_iter):
        if n_iter == 1:
            (y,x) = Images_test[:,:,0].shape
            img = Images_test_tensor
            image_initial = Images_test[:,:,(2,1,0)]
        else:
            (y,x) = Images_test[i][:,:,0].shape
            img = Images_test_tensor[i]
            image_initial = Images_test[i][:,:,(2,1,0)]
        
        
        img_tensor = torch.tensor(img).unsqueeze(0).permute(0,3,1,2)
        img_tensor = img_tensor.float()
        
        #img_tensor =  a_imgs
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        with torch.no_grad():
            output = model(img_tensor.to(device))[0]
            pred_mask = torch.argmax(output, axis=0)
        
        pred = pred_mask.squeeze().cpu().numpy()
        pred = np.uint8(pred*255)
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
    
        vista_1 = cv2.resize(pred, (x,y))
        vista_1 = vista_1[:,:,(2,1,0)]
        #other
        vista_2 = cv2.addWeighted(vista_1,0.4,image_initial,0.6,0)
        vista_3 = cv2.bitwise_and(image_initial, vista_1)
        
        filename_1 = '{}/{}_1.jpg'.format(file,names[i])
        filename_2 = '{}/{}_2.jpg'.format(file,names[i])
        filename_3 = '{}/{}_3.jpg'.format(file,names[i])
        # Using cv2.imwrite() method
        # Saving the image
        cv2.imwrite(filename_1, vista_1)
        cv2.imwrite(filename_2, vista_2)
        cv2.imwrite(filename_3, vista_3)
'''