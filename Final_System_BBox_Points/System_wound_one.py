# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 23:05:07 2022

@author: nicolas
"""

#import tool
import utils
import cv2 #Open CV Python
import numpy 
import numpy as np
import os
import detect
import generate_EDSR as gen_EDSR
from os import remove
import shutil
import tool_SVM
import tool
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



# the variable methos can be 1, 2, 3, 4

# 1: RGB
# 2: VGG
# 3: RGB and VGG
method = 3
name_experiment = 'VERRRR'
confiabilidad = 0.20


#instanciando
path_out_origin = "."
path_img = "generate_images_sr_img"
crops = 'crops'

#path yolov5 model
Path = '{}/{}'.format(path_out_origin, 'runs/train/exp/weights/best.pt')

#cargando las imagenes
ruta_images = 'images_one'
#ruta_masks = 'masks_one'
images = tool.encoder(ruta_images)

if isinstance(images, np.ndarray):  # Check if 'images' is a single numpy array
    images = [images]  # Convert the single image to a list of one image
#masks = tool.encoder(ruta_masks)



names_imgs = os.listdir(ruta_images)
if len(names_imgs) == 1:
    names_imgs = [names_imgs[0]]
names_imgs.sort()


dic_model = {1: 'RGB', 2: 'VGG', 3: 'RGB_VGG'}


ruta_carpeta = 'predicts_{}_{}'.format(dic_model[method], name_experiment)
tool.create_new_file(ruta_carpeta)
tool.create_new_file(crops)
iou_total = []

path_out = ruta_carpeta
tool.create_new_file('{}/{}'.format(ruta_carpeta,'overlay'))
tool.create_new_file('{}/{}'.format(ruta_carpeta,'crops_filter'))
tool.create_new_file('{}/{}'.format(ruta_carpeta,'marcas_crop'))
tool.create_new_file('{}/{}'.format(ruta_carpeta,'marcas_images'))
tool.create_new_file('{}/{}'.format(ruta_carpeta,'slic_boundaries'))
tool.create_new_file('{}/{}'.format(ruta_carpeta,'file_images_centroide_sample'))
tool.create_new_file('{}/{}'.format(ruta_carpeta,'images_bbox'))


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
        
        gen_EDSR.SR_EDSR(path_out_origin, 'crops')
        
        #img_with SR
        recorte_sr = cv2.imread('generate_images_sr_img/crop.jpg')
        
        shutil.copy('generate_images_sr_img/crop.jpg', '{}/{}/{}'.format(ruta_carpeta,'crops_filter', names_imgs[i] ))
        
        
        #guardamos lasss dimensiones
        h_crop_sr, w_crop_sr, _ = recorte_sr.shape
        
        # obtenemos los factores de escala
        fh = h_crop / h_crop_sr
        fw = w_crop/ w_crop_sr
        
        
        if (method == 1 or method == 2 or method == 3):
            n_segment = 100; compactness=10; sigma=1 ; threshold = 180; layer=5; file_per_image = True; file_iou_image = True
            input_point_all, input_label_all = tool_SVM.prediction_SVM_predict(path_out_origin, path_out, path_img,n_segment, compactness, sigma, threshold , layer, name_images, method, predictor, names_imgs[i])
        
       
        img_view = tool.show_points_on_image(recorte_sr.copy(), input_point_all, input_label_all)
        cv2.imwrite('{}/{}'.format('{}/{}'.format(ruta_carpeta,'marcas_crop'), names_imgs[i]), img_view)


        for coord in input_point_all:
            coord[0] = x1 + coord[0]*fw
            coord[1] = y1 + coord[1]*fh
         
        
        img_view_2 = tool.show_points_on_image(images[i].copy(), input_point_all, input_label_all, complete = True)
        cv2.imwrite('{}/{}'.format( '{}/{}'.format(ruta_carpeta,'marcas_images'),  names_imgs[i]), img_view_2[:,:,(2,1,0)])
        
        mask_pred = tool.segment_image(images[i], input_point_all, input_label_all,  bbox, predictor)
        masks_pred.append(mask_pred)
        
        
        
        img_bbox = tool.draw_bbox(img_view_2.copy(), bbox, color=(255, 0, 0))
        cv2.imwrite('{}/{}'.format('{}/{}'.format(ruta_carpeta,'images_bbox'), names_imgs[i]), img_bbox[:,:,(2,1,0)])
        
        
        
    if len(masks_pred) > 0:
        mask_predict_final = tool.unir_mascaras(masks_pred)
        mask_predict_final = np.squeeze(mask_predict_final)
        mask_predict_final_gray = mask_predict_final.astype(np.uint8) * 255
        
    mask_predict_final = cv2.cvtColor(mask_predict_final_gray, cv2.COLOR_GRAY2RGB)
    
    
    img_overlay = tool.overlay_mask(images[i], mask_predict_final_gray)
    

    
    cv2.imwrite('{}/{}/{}'.format(ruta_carpeta, 'overlay',names_imgs[i]), img_overlay[:,:,(2,1,0)])
    cv2.imwrite('{}/{}'.format(ruta_carpeta,names_imgs[i]), mask_predict_final)
    #iou = tool.calculate_iou(mask_predict_final, masks[i])
    #iou_total.append(iou)
    # Borrar el archivo
    os.remove(dst)
    os.remove('crops/crop.jpg')



