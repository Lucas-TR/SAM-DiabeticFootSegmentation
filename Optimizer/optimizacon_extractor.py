# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:23:44 2024

@author: nicol
"""
import cProfile
import psutil


import tool
import numpy as np
import pandas as pd
import cv2
from keras.applications.vgg16 import VGG16
import generate_feature_map as gen_vgg16

from skimage.segmentation import  slic
from skimage.segmentation import mark_boundaries
import os
import matplotlib.pyplot as plt

def feature_slic_prueba( n_segment, compactness, sigma, threshold, layer, path_img):
    model = VGG16()
    # Preparación de la imagen y carga de datos
    y_val = tool.encoder(path_img) if isinstance(path_img, str) else path_img
    n_iter = 1 if len(y_val.shape) == 3 else len(y_val)

    # Preparación de nombres de archivos y variables
    df_rgb, df_vgg = [], []

    for i in range(n_iter):
        img = y_val if n_iter == 1 else y_val[i]
        segments_slic = slic(img, n_segment, compactness=compactness, sigma=sigma)
        segments_slic = segments_slic - 1 if segments_slic[0][0] != 0 else segments_slic
        masks, positions_pxl = tool.create_masks_optimized(img, segments_slic) #optimizado
        x, y = tool.centroide(masks, segments_slic)
        channels = img.shape[2]
        var_name = sorted(os.listdir(path_img))
        name = [var_name[i]] * np.shape(x)[0]
        name_id = name[0].split('.')

        # Procesamiento para RGB
        var_ch_img_rgb, me_ch_img_rgb = extract_features_optimized(img, channels, masks, positions_pxl, segments_slic)

        # Procesamiento para VGG
        img_vgg = gen_vgg16.features_maps_VGG16(model, img, name_id[0], layer)

        channels_vgg = img_vgg.shape[2]
        var_ch_img_vgg, me_ch_img_vgg = extract_features_optimized(img_vgg, channels_vgg, masks, positions_pxl, segments_slic)

        # Agregar a listas
        df_rgb.append(create_df_optimized(var_ch_img_rgb, me_ch_img_rgb, name, x, y))
        df_vgg.append(create_df_optimized(var_ch_img_vgg, me_ch_img_vgg, name, x, y))

    # Concatenación y guardado de datos
    df_rgb = pd.concat(df_rgb, ignore_index=True)
    df_vgg = pd.concat(df_vgg, ignore_index=True).drop(['N_img', 'x_c', 'y_c'], axis='columns')
    df_complete = pd.concat([df_rgb, df_vgg], axis=1)
    #save_data(df_complete, name_experiment, path_out, vector_feature_all, exp_all)
       
    
    return df_complete

def extract_features_optimized(img, channels, masks, positions_pxl, segments_slic):
    var_ch_img = [None] * channels
    me_ch_img = [None] * channels
    
    for ch in range(channels):
        var_ch_img[ch], me_ch_img[ch] = tool.feature_color(img, ch, masks, positions_pxl, segments_slic)

    return var_ch_img, me_ch_img

def create_df_optimized(var_ch_img, me_ch_img, names, x_c, y_c):
    length = len(var_ch_img)
    var_all = [None] * length
    me_all = [None] * length
    
    for i in range(length):
        var_all[i] = var_ch_img[i]
        me_all[i] = me_ch_img[i]

    return tool.create_df_feature_color_optimized(var_all, me_all, names, x_c, y_c, bool_label=False)



def save_data(df_complete, name_experiment, path_out, vector_feature_all, exp_all):
    PATH = path_out
    file_name = f"{name_experiment}_feature_vector_RGB-VGG"
    df_complete.to_csv(f"{PATH}/{name_experiment}/{vector_feature_all}/{file_name}.csv")
    df_complete.to_excel(f"{PATH}/{name_experiment}/{vector_feature_all}/{file_name}.xlsx")
    if exp_all:
        df_complete.to_csv(f"{PATH}/Data_training/{file_name}.csv")
        

n_segment =100; compactness=10; sigma=1; threshold=180;layer =5

path_img = r'C:\Users\nicol\OneDrive\Escritorio\Github\SAM-DiabeticFootSegmentation\Optimizer\img_test'

cpu_antes = psutil.cpu_percent()


ver = feature_slic_prueba( n_segment, compactness, sigma, threshold, layer, path_img)

cpu_despues = psutil.cpu_percent()