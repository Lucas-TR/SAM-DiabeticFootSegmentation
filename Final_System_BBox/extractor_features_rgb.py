# -*- coding: utf-8 -*-
"""Copy of Extractor_caracteristicas_2 - VRI.ipynb
### **Funciones**
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import  slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import shutil
from scipy import stats
from scipy import ndimage
import random
from keras.applications.vgg16 import VGG16
import generate_feature_map as gen_vgg16

import tool


#Función para extraer  caracteristicas

# creamos la mascara para cada superpixel
#creando tabla de datos para el entrenamiento
#n_segment = 150; threshold = 180
#empty = [11,52,160] # mascaras vacias, se tienen que observar esas imágenes

def feature_slic(name_experiment, n_segment , compactness, sigma, threshold, path_img , path_mask, path_out, boolean, boolean_2, exp_all=False):
  RGB=True
  # load the model
  #model = VGG16()
  if (isinstance(path_img, str) and isinstance(path_mask, str)):
      y_val = tool.encoder(path_img)
      y_val_mask = tool.encoder(path_mask)
  else:
      y_val= path_img
      y_val_mask = path_mask
  file_data_per_image = 'vector_feature_per_image'
  file_boundaries_rgb = 'Boundaries_rgb'
  file_feature_all = 'vector_feature_all'
  file_test_iou_per_image = 'Labels'
  file_vgg16_image = 'Images_vgg16'
  file_vgg16_slic_image = 'Images_vgg16_slic'
  file_vgg16_slic_boundaries = 'Images_vgg16_slic_boundaries'
  vector_feature_all = 'vector_feature_all'
  
  
  var_ch_img= []; me_ch_img=[]; as_ch_img=[]; f_ch_img=[]; i_ch_img=[]
  label =[] ; names = [] ; cont_pxls =[]; X_c=[]; Y_c=[]

  if len(y_val.shape) == 3:
     n_iter = 1
  else:
     n_iter = len(y_val)

  for i in range(n_iter):
      #creating a file in which the results will be stored
      tool.new_file(path_out,name_experiment,i)
  #  if not(i in empty):
      if (n_iter==1):
        img = y_val
        mask_ref = y_val_mask
        #channels = y_val.shape[2] #number of channels
      else:
        img = y_val[i]
        mask_ref = y_val_mask[i]
          
      segments_slic = slic(img, n_segment , compactness=compactness , sigma = sigma)
      if segments_slic[0][0] != 0:
          segments_slic = segments_slic - 1 #segments_slic has to start at 0
      cont_pxl = tool.contador_pxls(segments_slic)
      masks, positions_pxl = tool.create_masks(img,segments_slic)
      x, y = tool.centroide(masks, segments_slic)
      img_rgb = img
      
      #la imagen debe estar en formato RGB
      clas, wound, ctd_s_pxl = tool.detector(mask_ref, positions_pxl,segments_slic,threshold)
      name = []
      
      for k in range(np.shape(wound)[0]):
            var_name = os.listdir(path_img) 
            var_name.sort()
            name.append(var_name[i])
      name_id = name[0].split('.')
      
      if(not RGB):
          tool.new_file(path_out,'{}/{}'.format(name_experiment, file_vgg16_image),i)
          
          #img = gen_vgg16.features_maps_VGG16(model, img, '{}/{}'.format(name_experiment, file_vgg16_image) , name_id[0], layer)
          plt.close()

      channels = img.shape[2] #number of channels
      
      for ch in range(channels):
          var_ch_sp, me_ch_sp, as_ch_sp, f_ch_sp, i_ch_sp = tool.feature_color(img, ch,masks,positions_pxl, segments_slic) #channel red
          var_ch_img.append(var_ch_sp) ; me_ch_img.append(me_ch_sp); as_ch_img.append(as_ch_sp); f_ch_img.append(f_ch_sp); i_ch_img.append(i_ch_sp)
      
      if (boolean):
          var_ch_img_aux = var_ch_img[-channels:]; me_ch_img_aux = me_ch_img[-channels:]; as_ch_img_aux=as_ch_img[-channels:]; f_ch_img_aux=f_ch_img[-channels:]; i_ch_img_aux=i_ch_img[-channels:]
          
          df_aux = tool.create_df_feature_color(var_ch_img_aux, me_ch_img_aux, as_ch_img_aux, f_ch_img_aux, i_ch_img_aux, wound, name, cont_pxl, x, y)
          
          tool.new_file(path_out,'{}/{}'.format(name_experiment,file_data_per_image),i)
          PATH = path_out
          df_aux.to_csv("{}/{}/{}/{}_{}_RGB.csv".format(PATH,name_experiment,file_data_per_image,name_experiment,name_id[0]))
          #df_aux.to_excel("{}/{}/feature_vector_{}.xlsx".format(PATH,file,i))
          
      if(boolean_2):
          tool.new_file(path_out,'{}/{}'.format(name_experiment,file_test_iou_per_image),i)
          path_save = "{}/{}/{}".format(path_out,name_experiment, file_test_iou_per_image)
          tool.create_img(img_rgb, mask_ref, segments_slic, clas,path_save,wound,name_id[0])
          plt.close()
          tool.new_file(path_out,'{}/{}'.format(name_experiment,file_boundaries_rgb),i)
          path_save = "{}/{}/{}".format(path_out,name_experiment, file_boundaries_rgb)
          
          img_save = mark_boundaries(img_rgb , segments_slic)
          plt.imshow(img_save)
          plt.axis(False)
          plt.savefig("{}/{}.jpg".format(path_save, name_id[0]))
          
          
          
          if(not RGB):
              tool.new_file(path_out,'{}/{}'.format(name_experiment,file_vgg16_slic_image),i)
              path_save = "{}/{}/{}".format(path_out,name_experiment, file_vgg16_slic_image)    
              file_per_image = "{}/{}".format(path_save,name_id[0])
              os.mkdir(file_per_image)
              
              tool.new_file(path_out,'{}/{}'.format(name_experiment,file_vgg16_slic_boundaries),i)
              path_save = "{}/{}/{}".format(path_out,name_experiment, file_vgg16_slic_boundaries)
              file_per_boundaries = "{}/{}".format(path_save,name_id[0])
              os.mkdir(file_per_boundaries)
              
              
              n_images = 5
              for _ in range(n_images):
                  ch = random.randint(0, channels-1) 
                  img_ch =  np.array(img[:,:,ch], dtype=np.uint8)
                  norm = plt.Normalize(img_ch .min(), img_ch .max())
                  numpy_array = plt.cm.jet(norm(img_ch))
                  img_ch = numpy_array[:,:,0:3]
                  
                  #save boundaries
                  img_save = mark_boundaries(img_ch , segments_slic)
                  plt.imshow(img_save)
                  plt.title("Channel: {}".format(ch))
                  plt.axis(False)
                  plt.savefig("{}/{}.jpg".format(file_per_boundaries, name_id[0]+str(ch+1)))
                  #save comparation
                  plt.close()
                  tool.create_img(img_ch, mask_ref, segments_slic, clas,file_per_image,wound,name_id[0]+str(ch+1))
                  plt.close()
              
      #if (n_iter==1):
       # tool.create_img(img_rgb, mask_ref, segments_slic, clas,path_out,wound,name_id[0])
        
      label.append(wound); names.append(name) ; cont_pxls.append(cont_pxl); X_c.append(x); Y_c.append(y) 
  
  #create data frame
  var_all =[]; me_all =[]; as_all = [] ; f_all = [] ; i_all = []
  for i in range(channels):
      var_imgs =[]; me_imgs =[]; as_imgs = [] ; f_imgs = [] ; i_imgs = []
      for j in range(n_iter):
          var_imgs.append(var_ch_img[i+channels*j])
          me_imgs.append(me_ch_img[i+channels*j])
          as_imgs.append(as_ch_img[i+channels*j])
          f_imgs.append(f_ch_img[i+channels*j])
          i_imgs.append(i_ch_img[i+channels*j])
        
      var_all.append(tool.flatten(var_imgs)) ; me_all.append(tool.flatten(me_imgs)); as_all.append(tool.flatten(as_imgs)); f_all.append(tool.flatten(f_imgs)); i_all.append(tool.flatten(i_imgs))


  label =  tool.flatten(label) ; names = tool.flatten(names) ; cont_pxls = tool.flatten(cont_pxls);  x_c = tool.flatten(X_c); y_c = tool.flatten(Y_c) 


  df = tool.create_df_feature_color(var_all,me_all,as_all,f_all,i_all,label,names,cont_pxls,x_c,y_c)
  
  
  tool.new_file(path_out,'{}/{}'.format(name_experiment,vector_feature_all),0)

  PATH = path_out
  df.to_csv("{}/{}/{}/{}_feature_vector_RGB.csv".format(PATH,name_experiment,vector_feature_all,name_experiment))
  df.to_excel("{}/{}/{}/{}_feature_vector_RGB.xlsx".format(PATH,name_experiment,vector_feature_all,name_experiment))
  if exp_all:
      df.to_csv("{}//{}/{}_feature_vector_RGB.csv".format(PATH,'Data_training',name_experiment))
  '''
  <---- End --->
  '''
  name = 'README.txt'
  README = '{}/{}/{}'.format(PATH,name_experiment,name)
  names = ['Name Experiment', 'Images','Numbers of channels','n_segment','compactness', 'sigma', 'threshold', 'file_per_image', 'file_iou_image']
  inputs = [name_experiment, n_iter, channels,n_segment , compactness, sigma, threshold, boolean, boolean_2]
  with open (README, 'w') as f:
    f.write('Experiment performed using RGB\n\n INPUTS \n\n')
    for idx in range(len(inputs)):
        f.write('{} : {} \n'.format(names[idx],str(inputs[idx])))
    f.close()

  print("Finish...!")
  return df

def feature_slic_prueba(name_experiment, n_segment , compactness, sigma, threshold, path_img, path_out, boolean, boolean_2, exp_all=False):
  RGB=True
  # load the model
  #model = VGG16()
  if (isinstance(path_img, str)):
      y_val = tool.encoder(path_img)
  else:
      y_val= path_img

  file_data_per_image = 'vector_feature_per_image'
  file_boundaries_rgb = 'Boundaries_rgb'
  file_feature_all = 'vector_feature_all'
  file_test_iou_per_image = 'Labels'
  file_vgg16_image = 'Images_vgg16'
  file_vgg16_slic_image = 'Images_vgg16_slic'
  file_vgg16_slic_boundaries = 'Images_vgg16_slic_boundaries'
  vector_feature_all = 'vector_feature_all'
  
  
  var_ch_img= []; me_ch_img=[]; as_ch_img=[]; f_ch_img=[]; i_ch_img=[]
  names = [] ; cont_pxls =[]; X_c=[]; Y_c=[]
  

  if len(y_val.shape) == 3:
     n_iter = 1
  else:
     n_iter = len(y_val)

  for i in range(n_iter):
      #creating a file in which the results will be stored
      tool.new_file(path_out,name_experiment,i)
  #  if not(i in empty):
      if (n_iter==1):
        img = y_val
        #channels = y_val.shape[2] #number of channels
      else:
        img = y_val[i]
          
      segments_slic = slic(img, n_segment , compactness=compactness , sigma = sigma)
      if segments_slic[0][0] != 0:
          segments_slic = segments_slic - 1 #segments_slic has to start at 0
      cont_pxl = tool.contador_pxls(segments_slic)
      masks, positions_pxl = tool.create_masks(img,segments_slic)
      x, y = tool.centroide(masks, segments_slic)
      img_rgb = img
      
      #la imagen debe estar en formato RGB

      name = []
      
      for k in range(np.shape(x)[0]):
            var_name = os.listdir(path_img) 
            var_name.sort()
            name.append(var_name[i])
      name_id = name[0].split('.')
      

      channels = img.shape[2] #number of channels
      
      for ch in range(channels):
          var_ch_sp, me_ch_sp, as_ch_sp, f_ch_sp, i_ch_sp = tool.feature_color(img, ch,masks,positions_pxl, segments_slic) #channel red
          var_ch_img.append(var_ch_sp) ; me_ch_img.append(me_ch_sp); as_ch_img.append(as_ch_sp); f_ch_img.append(f_ch_sp); i_ch_img.append(i_ch_sp)
      
      if (boolean):
          var_ch_img_aux = var_ch_img[-channels:]; me_ch_img_aux = me_ch_img[-channels:]; as_ch_img_aux=as_ch_img[-channels:]; f_ch_img_aux=f_ch_img[-channels:]; i_ch_img_aux=i_ch_img[-channels:]
          
          df_aux = tool.create_df_feature_color(var_ch_img_aux, me_ch_img_aux, as_ch_img_aux, f_ch_img_aux, i_ch_img_aux, name, cont_pxl, x, y,bool_label=False)
          
          tool.new_file(path_out,'{}/{}'.format(name_experiment,file_data_per_image),i)
          PATH = path_out
          df_aux.to_csv("{}/{}/{}/{}_RGB.csv".format(PATH,name_experiment,file_data_per_image,name_experiment))
          #df_aux.to_excel("{}/{}/feature_vector_{}.xlsx".format(PATH,file,i))
                 
      #if (n_iter==1):
       # tool.create_img(img_rgb, mask_ref, segments_slic, clas,path_out,wound,name_id[0])
        
      names.append(name) ; cont_pxls.append(cont_pxl); X_c.append(x); Y_c.append(y) 
  
  #create data frame
  var_all =[]; me_all =[]; as_all = [] ; f_all = [] ; i_all = []
  for i in range(channels):
      var_imgs =[]; me_imgs =[]; as_imgs = [] ; f_imgs = [] ; i_imgs = []
      for j in range(n_iter):
          var_imgs.append(var_ch_img[i+channels*j])
          me_imgs.append(me_ch_img[i+channels*j])
          as_imgs.append(as_ch_img[i+channels*j])
          f_imgs.append(f_ch_img[i+channels*j])
          i_imgs.append(i_ch_img[i+channels*j])
        
      var_all.append(tool.flatten(var_imgs)) ; me_all.append(tool.flatten(me_imgs)); as_all.append(tool.flatten(as_imgs)); f_all.append(tool.flatten(f_imgs)); i_all.append(tool.flatten(i_imgs))


  names = tool.flatten(names) ; cont_pxls = tool.flatten(cont_pxls);  x_c = tool.flatten(X_c); y_c = tool.flatten(Y_c) 


  df = tool.create_df_feature_color(var_all,me_all,as_all,f_all,i_all,names,cont_pxls,x_c,y_c,bool_label=False)
  
  
  tool.new_file(path_out,'{}/{}'.format(name_experiment,vector_feature_all),0)

  PATH = path_out
  df.to_csv("{}/{}/{}/{}_feature_vector_RGB.csv".format(PATH,name_experiment,vector_feature_all,name_experiment))
  df.to_excel("{}/{}/{}/{}_feature_vector_RGB.xlsx".format(PATH,name_experiment,vector_feature_all,name_experiment))
  if exp_all:
      df.to_csv("{}//{}/{}_feature_vector_RGB.csv".format(PATH,'Data_training',name_experiment))
  '''
  <---- End --->
  '''
  name = 'README.txt'
  README = '{}/{}/{}'.format(PATH,name_experiment,name)
  names = ['Name Experiment', 'Images','Numbers of channels','n_segment','compactness', 'sigma', 'threshold', 'file_per_image', 'file_iou_image']
  inputs = [name_experiment, n_iter, channels,n_segment , compactness, sigma, threshold, boolean, boolean_2]
  with open (README, 'w') as f:
    f.write('Experiment performed using RGB\n\n INPUTS \n\n')
    for idx in range(len(inputs)):
        f.write('{} : {} \n'.format(names[idx],str(inputs[idx])))
    f.close()

  print("Finish...!")
  return df