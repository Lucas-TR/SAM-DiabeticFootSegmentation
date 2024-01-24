# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:36:49 2022

@author: nicol
"""
import unet
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tool


#almacenando y cargando el modelo
ruta_test = "images_test"
ruta_mask_test = "masks_test"

path_out = "."

#Definimos ruta donde esta el modelo guardado
ruta_model = '{}/{}'.format('model_UNET','model_4')

#Cargamos el modelo entrenado Uet
model = torch.load('{}/{}'.format(ruta_model,'model.pt'))

#Cargamos las imagenes en forma de tensor
Images_test_tensor = unet.encoder_images(ruta_test)
Images_mask_test_tensor = unet.encoder_masks(ruta_mask_test)

#Cargamos imágenes en forma de arreglos
Images_test = tool.encoder(ruta_test)
Images_mask_test = tool.encoder(ruta_mask_test)

#Extraemos los nombres de las imagenes
names_dot = tool.encoder_files(ruta_test)
names = tool.split_dot(names_dot)


    
#Creamos las carpetas en donde guardaremos las predicciones
file = 'Predictions_unet_final'
file_grab = 'Predictions_unet_grab_final'

tool.new_file(path_out, file, 0)
tool.new_file(path_out, file_grab, 0)

new_path_out = '{}/{}'.format(path_out,file)
new_path_out_grab = '{}/{}'.format(path_out,file_grab)


tool.new_file(new_path_out, 'all', 0)
tool.new_file(new_path_out_grab, 'all', 0)

new_path_out_final = '{}/{}'.format(new_path_out,'all')
new_path_out_grab_final = '{}/{}'.format(new_path_out_grab,'all')

#Creamos los arreglos en donde almacenaremos la metrica iou de las predicciones
IOU_prom_grap =[]; IOU_prom =[]

for i in range(len(Images_test_tensor)):
    
    #guardamos la forma original de la imágen
    (y,x) = Images_test[i][:,:,0].shape
    
    #imagen de entrada
    img = Images_test_tensor[i]
    img_tensor = torch.tensor(img).unsqueeze(0).permute(0,3,1,2)
    img_tensor = img_tensor.float()
    
    #Realizamos la predicción
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.to(device))[0]
        pred_mask = torch.argmax(output, axis=0)
    
    pred = pred_mask.squeeze().cpu().numpy()
    pred = np.uint8(pred*255)
    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
    
    #Cambiamos el tamaño de la predicción al tamaño inicial de la impagen
    pred_re = cv2.resize(pred, (x,y))
    iou = tool.calculate_iou(Images_mask_test[i], pred_re)
    plt.imshow(pred_re)
    plt.title('IOU = {:.2f}'.format(iou))
    plt.axis(False)
    plt.savefig("{}/{}.jpg".format(new_path_out_final,names[i]))
    plt.close()
    IOU_prom.append(iou)
    
    tool.new_file(new_path_out,names[i] , 0)
    tool.create_img_unet(Images_test[i], Images_mask_test[i], pred_re, '{}/{}'.format(new_path_out,names[i]), names[i])
    
    #Esta parte se probo el metodo GRABcut para la mejora de las predcciones, pero por ahora no se obtuvo buenos resultados.
    new_pred = tool.grub_cut(Images_test[i], pred_re)
    iou = tool.calculate_iou(Images_mask_test[i], new_pred)
    plt.imshow(new_pred)
    plt.title('IOU = {:.2f}'.format(iou))
    plt.axis(False)
    plt.savefig("{}/{}.jpg".format(new_path_out_grab_final,names[i]))
    plt.close()
    IOU_prom_grap.append(iou)
    
    tool.new_file(new_path_out_grab,names[i] , 0)
    tool.create_img_unet(Images_test[i], Images_mask_test[i], new_pred, '{}/{}'.format(new_path_out_grab,names[i]), names[i])

#Guardemos los resultados del etrenamiento en un archivo txt
name = 'README.txt'
README = '{}/{}'.format(path_out,name)

with open (README, 'w') as f:
   f.write('Performance: \n\n')
   f.write('IOU_prom = {:.2f} \n IOU_prom_grab = {:.2f} \n'.format(sum(IOU_prom)/len(IOU_prom),sum(IOU_prom_grap)/len(IOU_prom_grap)))
   f.close()
print("Finish...!")    
