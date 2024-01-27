# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 18:45:12 2023

@author: nicol
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from scipy.io import savemat

def calcular_metricas(ruta_etiquetas, ruta_predicciones, umbral=50, save_overlay = False):
    archivos_etiquetas = sorted(os.listdir(ruta_etiquetas))
    archivos_predicciones = sorted(os.listdir(ruta_predicciones))

    resultados = []
    for etq_archivo, pred_archivo in tqdm(zip(archivos_etiquetas, archivos_predicciones), total=len(archivos_etiquetas), desc="Calculando métricas"):
        etq_img = Image.open(os.path.join(ruta_etiquetas, etq_archivo)).convert('L')
        pred_img = Image.open(os.path.join(ruta_predicciones, pred_archivo)).convert('L')

        etq_arr = np.array(etq_img)
        pred_arr = np.array(pred_img)

        # Binariza las imágenes usando el umbral
        etq_arr = np.where(etq_arr < umbral, 0, 1)
        pred_arr = np.where(pred_arr < umbral, 0, 1)

        
        interseccion = np.logical_and(etq_arr, pred_arr)
        union = np.logical_or(etq_arr, pred_arr)
        
        if save_overlay:
            # Crea una imagen de los resultados
            resultado_img = np.zeros_like(etq_arr, dtype=np.uint8)
            resultado_img = np.stack([resultado_img]*3, axis=-1)  # Convierte a RGB
            resultado_img[interseccion] = [0, 255, 0]  # Verdaderos positivos en verde
            resultado_img[np.logical_and(etq_arr, np.logical_not(pred_arr))] = [0, 0, 255]  # Falsos negativos en azul
            resultado_img[np.logical_and(np.logical_not(etq_arr), pred_arr)] = [255, 0, 0]  # Falsos positivos en rojo
            resultado_img[np.logical_and(np.logical_not(etq_arr), np.logical_not(pred_arr))] = [128, 128, 128]  # Verdaderos negativos en gris
            resultado_img = Image.fromarray(resultado_img)
            resultado_img.save(os.path.join(ruta_predicciones, f"resultado_{pred_archivo}"))
            
            # Guarda la imagen en formato .mat
            savemat(os.path.join(ruta_predicciones, f"resultado_{pred_archivo}.mat"), {'resultado_img': np.array(resultado_img)})
            
        iou = np.sum(interseccion) / np.sum(union)
        dice = 2. * np.sum(interseccion) / (np.sum(etq_arr) + np.sum(pred_arr))
        accuracy = accuracy_score(etq_arr.flatten(), pred_arr.flatten())
        precision = precision_score(etq_arr.flatten(), pred_arr.flatten())
        recall = recall_score(etq_arr.flatten(), pred_arr.flatten())
        f1 = f1_score(etq_arr.flatten(), pred_arr.flatten())

        resultados.append([etq_archivo, iou, dice, accuracy, precision, recall, f1])

    df_resultados = pd.DataFrame(resultados, columns=['nombre_imagen', 'IoU', 'Dice', 'Accuracy', 'Precision', 'Recall', 'F1'])

    print(f"IoU promedio: {df_resultados['IoU'].mean()}")
    print(f"Coeficiente de Dice promedio: {df_resultados['Dice'].mean()}")
    print(f"Accuracy promedio: {df_resultados['Accuracy'].mean()}")
    print(f"Precision promedio: {df_resultados['Precision'].mean()}")
    print(f"Recall promedio: {df_resultados['Recall'].mean()}")
    print(f"F1 score promedio: {df_resultados['F1'].mean()}")

    return df_resultados





def top_iou(df_resultados, top_n=100):
    return df_resultados.nlargest(top_n, 'IoU')



def comparar_resultados(df1, df2):
    # Set 'Times New Roman' as font
    plt.rcParams["font.family"] = "Times New Roman"
    
    # Calcula las medias de las columnas de interés
    medias_df1 = df1[['IoU', 'Dice', 'Accuracy', 'Precision', 'Recall', 'F1']].mean()
    medias_df2 = df2[['IoU', 'Dice', 'Accuracy', 'Precision', 'Recall', 'F1']].mean()

    # Define una paleta de colores personalizada
    my_cmap = ListedColormap(['#800000', '#FFA500'])  # Ginda y anaranjado claro

    # Width of the bars and the space between bar groups
    bar_width = 0.2  # decreased from 0.35 to 0.2
    space_width = 0.05

    # Position of the bars
    r1 = np.arange(len(medias_df1))
    r2 = [x + bar_width + space_width for x in r1]

    # Create the figure
    fig, ax = plt.subplots()

    # Add the bars
    ax.bar(r1, medias_df1, color=my_cmap(0), width=bar_width, edgecolor='grey', label='SAM: box + points')
    ax.bar(r2, medias_df2, color=my_cmap(1), width=bar_width, edgecolor='grey', label='SAM: box')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add some details to the plot
    ax.set_xticks([r + bar_width / 2 for r in range(len(medias_df1))])
    ax.set_xticklabels(medias_df1.index)
    ax.set_ylim([0, 1])
    
    # Add the legend to the lower right corner
    ax.legend(loc='lower right')

    # Show the plot
    plt.xticks(rotation='horizontal')

    # Save the figure
    plt.savefig('comparacion_resultados.png', dpi=300)


# Uso de la función:
ruta_etiquetas = r'D:\PROYECTS\Monitoring_of_Wound\Final_System_BBox_Points\masks'
ruta_predicciones_bbox = r'C:\Users\nicol\OneDrive\Escritorio\Github\SAM-DiabeticFootSegmentation\Final_System_BBox\exp_final_fine_tunin_bbox\masks'

ruta_predicciones_bbox_points = r'C:\Users\nicol\OneDrive\Escritorio\Github\SAM-DiabeticFootSegmentation\Final_System_BBox_Points\predicts_RGB_VGG_experiment_final_fine_tunin\masks'

ruta_predicciones_paper= r'D:\PROYECTS\Monitoring_of_Wound\Final_System_BBox_Points\experiments\predicts_RGB_VGG_paper'

ruta_predicciones_points = r'C:\Users\nicol\OneDrive\Escritorio\Github\SAM-DiabeticFootSegmentation\Final_System_Points\exp_final_fine_tunin_pointsRGB_VGG\masks'

df_resultados= calcular_metricas(ruta_etiquetas, ruta_predicciones_bbox)


df_resultados2 = calcular_metricas(ruta_etiquetas, ruta_predicciones_paper)

df_resultados3 = calcular_metricas(ruta_etiquetas, ruta_predicciones_bbox_points)

df_resultados4 = calcular_metricas(ruta_etiquetas, ruta_predicciones_points)



df_top_iou = top_iou(df_resultados2, top_n=83)

nombres_repetidos = df_top_iou['nombre_imagen'].unique()
df_top_iou2 = df_resultados2[df_resultados2['nombre_imagen'].isin(nombres_repetidos)]

z_mean1 = df_top_iou['IoU'].mean()

z_mean2 = df_top_iou2['IoU'].mean()

comparar_resultados(df_top_iou, df_top_iou2)

df_top_iou.to_excel('results_box_points.xlsx', index=False)
df_top_iou2.to_excel('results_SAM_200.xlsx', index=False)


'''
iou_scores_bbox, dice_scores_bbox, accuracy_scores_bbox = calcular_metricas(ruta_etiquetas, ruta_predicciones_bbox)



# Obtiene los índices de los elementos en lista1 que son mayores que 0.80
indices = [i for i, x in enumerate(iou_scores) if x > 0.75]

# Usa los índices para obtener una sublista de lista2
sublista_bp = [iou_scores[i] for i in indices]
sublista_b = [iou_scores_bbox[i] for i in indices]

print(np.mean(sublista_bp))

print(np.mean(sublista_b))
'''


