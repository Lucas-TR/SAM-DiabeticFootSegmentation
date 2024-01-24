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

#Función para extraer  caracteristicas

# creamos la mascara para cada superpixel
#creando tabla de datos para el entrenamiento
#n_segment = 150; threshold = 180
#empty = [11,52,160] # mascaras vacias, se tienen que observar esas imágenes

def feature_slic(n_segment , compactness, sigma, threshold , path_img , path_mask, path_out, boolean):

  y_val = encoder(path_img)
  y_val_mask = encoder(path_mask)

  var_r= [] ; me_r = [] ; as_r = []; f_r = []; i_r = []
  var_g= [] ; me_g = [] ; as_g = []; f_g = []; i_g = []
  var_b= [] ; me_b = [] ; as_b = []; f_b = []; i_b = []
  label =[] ; names = [] ; cont_pxls =[]; X_c=[]; Y_c=[]
  file = 'Data_per_imge'
  if len(y_val.shape) == 3:
    n_iter = 1
  else:
    n_iter = len(y_val)

  for i in range(n_iter):
  #  if not(i in empty):
      if (n_iter==1):
        img = y_val
        mask_ref = y_val_mask
      else:
        img = y_val[i]
        mask_ref = y_val_mask[i]

      segments_slic = slic(img, n_segment , compactness=compactness , sigma = sigma)
      cont_pxl = contador_pxls(segments_slic)
      masks, positions_pxl = create_masks(img,segments_slic)
      x, y = centroide(masks, segments_slic)
      #la imagen debe estar en formato RGB
      # var: variance
      # me: mean
      # as: assymetry
      # f: maximum frequency: color/value that repeats the most
      # i: maximum intensity
      # other possible are texture with entropy
      # from a paper: get source for citation Automatic measurement of pressure ulcers using Support Vector Machines and GrabCut

      var_0, me_0, as_0, f_0, i_0 = feature_color(img,0,masks,positions_pxl, segments_slic) #channel red
      var_1, me_1, as_1, f_1, i_1 = feature_color(img,1,masks,positions_pxl, segments_slic) #channel green
      var_2, me_2, as_2, f_2, i_2 = feature_color(img,2,masks,positions_pxl, segments_slic) #channel blue

      # make the matching of the class with the superpixels
      clas, wound, ctd_s_pxl = detector(mask_ref, positions_pxl,segments_slic,threshold)
      name = []
      for k in range(np.shape(wound)[0]):
            var_name = os.listdir(path_img)
            var_name.sort()
            name.append(var_name[i])
      if (boolean):
          var_aux= [] ; me_aux = [] ; as_aux = []; f_aux = []; i_aux = []

          var_aux.append(var_0) ; me_aux.append(me_0); as_aux.append(as_0); f_aux.append(f_0); i_aux.append(i_0)
          var_aux.append(var_1) ; me_aux.append(me_1); as_aux.append(as_1); f_aux.append(f_1); i_aux.append(i_1)
          var_aux.append(var_2) ; me_aux.append(me_2); as_aux.append(as_2); f_aux.append(f_2); i_aux.append(i_2)

          df_aux = create_df_feature_color(var_aux,me_aux,as_aux,f_aux,i_aux,wound,name,cont_pxl,x,y)

          files = os.listdir('.')
          if file in files and i==0:
              shutil.rmtree(file)
              os.mkdir(file)
          else:
              if i==0:
                  os.mkdir(file)
          PATH = path_out
          name_id = name[0].split('.')

          df_aux.to_csv("{}/{}/{}.csv".format(PATH,file,name_id[0]))
          #df_aux.to_excel("{}/{}/feature_vector_{}.xlsx".format(PATH,file,i))

      # this is for plotting and only works when there is only one image as input
      if (n_iter==1):
        mask_sp, mask_edge_sp, mask_edge_manual, img_comparative = labels_visual(img, mask_ref, segments_slic, clas)

        images = [mask_sp, mask_ref, mask_edge_sp, mask_edge_manual]
        labels = ['mask SP', 'mask Manual', 'Edge SP', 'Edge Manual']
        save_img(images,labels,path_out,wound)

      var_r.append(var_0) ; me_r.append(me_0); as_r.append(as_0); f_r.append(f_0); i_r.append(i_0)
      var_g.append(var_1) ; me_g.append(me_1); as_g.append(as_1); f_g.append(f_1); i_g.append(i_1)
      var_b.append(var_2) ; me_b.append(me_2); as_b.append(as_2); f_b.append(f_2); i_b.append(i_2)
      label.append(wound); names.append(name) ; cont_pxls.append(cont_pxl); X_c.append(x); Y_c.append(y) 

  #creando data frame
  var_r =  flatten(var_r) ; me_r = flatten(me_r); as_r = flatten(as_r); f_r = flatten(f_r); i_r = flatten(i_r)
  var_g =  flatten(var_g) ; me_g = flatten(me_g); as_g = flatten(as_g); f_g = flatten(f_g); i_g = flatten(i_g)
  var_b =  flatten(var_b) ; me_b = flatten(me_b); as_b = flatten(as_b); f_b = flatten(f_b); i_b = flatten(i_b)
  label =  flatten(label) ; names = flatten(names) ; cont_pxls = flatten(cont_pxls);  x_c = flatten(X_c); y_c = flatten(Y_c) 

  var_ =[]; me_ =[]; as_ = [] ; f_ = [] ; i_ = []
  var_.append(var_r) ; var_.append(var_g) ; var_.append(var_b) 
  me_.append(me_r) ; me_.append(me_g) ; me_.append(me_b) 
  as_.append(as_r) ; as_.append(as_g) ; as_.append(as_b)
  f_.append(f_r); f_.append(f_g); f_.append(f_b)
  i_.append(i_r); i_.append(i_g); i_.append(i_b)


  df = create_df_feature_color(var_,me_,as_,f_,i_,label,names,cont_pxls,x_c,y_c)

  files = os.listdir('.')
  if 'vector_feature' in files: shutil.rmtree("vector_feature");

  file = 'vector_feature'
  os.mkdir(file)

  PATH = path_out
  df.to_csv("{}/{}/feature_vector.csv".format(PATH,file))
  df.to_excel("{}/{}/feature_vector.xlsx".format(PATH,file))
  print("Finish...!")


def create_masks(img,segments_slic):
  superPixels = []; masks = []
  for i in range(np.max(segments_slic)):
    superPixels.append(np.where(segments_slic==i+1))
    mask = np.zeros(img.shape, dtype="uint8")
    mask[superPixels[i]] = 255
    masks.append(mask[:,:,0])
  return masks, superPixels

#Extraer caracteristicas de color para una sola imagen
def feature_color( img, ch, masks, sup_pxls, segments_slic):
  var_ch = []; me_ch = []; as_ch = []; f_ch = []; i_ch = []
  for i in range(np.max(segments_slic)):
    img_ch = img[:,:,ch][sup_pxls[i]]
    hist = cv2.calcHist([img],[ch],masks[i],[256],[0,256])
    var_ch.append(np.var(img_ch))
    me_ch.append(np.mean(img_ch))
    as_ch.append(stats.kurtosis(img_ch))
    f_ch.append(np.max(hist))
    i_ch.append(np.max(img_ch))
  return var_ch, me_ch, as_ch, f_ch, i_ch

def detector(mask_ref, superPixels, segments_slic, mean_min):
  clas = [] # se guardara el número de superpixel que contiene lesión
  wound = [] # Se etiquetara al superpixel
  for i in range(np.max(segments_slic)):
    if np.mean(mask_ref[superPixels[i]])>mean_min:
      clas.append(i)
      wound.append(1)
    else:
      wound.append(0)
  ctd_sup = len(clas) #cantidad de superpixeles necesarios para cubrir la lesión
  return clas, wound, ctd_sup

#función solo para crear mascara con superpixeles
def create_mask_sp(img,segments_slic,clas):
  marcadores = []
  for i in clas:
    marcadores.append(np.where(segments_slic==i))

  #creamos la máscara usando superpixeles
  mask_sp = np.zeros(img.shape, dtype="uint8")
  for idx in range(len(marcadores)):
    mask_sp[marcadores[idx]] = 255
  return mask_sp

#Función para extraer imagen por superpixeles y comparación
# Extraemos todos los pixeles que contienen la lesión
def labels_visual(img, mascara, segments_slic, clas):
  marcadores = []
  for i in clas:
    marcadores.append(np.where(segments_slic==i))

  #creamos la máscara usando superpixeles
  mask_sp = np.zeros(img.shape, dtype="uint8")
  for idx in range(len(marcadores)):
    mask_sp[marcadores[idx]] = 255

  #contorno de mascara con sp
  S = cv2.Canny(mask_sp,250,255)

  #contorno de mascara
  Q = cv2.Canny(mascara,250,255)

  #Graficando contorno de mascara manual
  mask_edge_manual = mark_boundaries(img, segments_slic)[:,:,(2,1,0)]
  (N,M) = img[:,:,0].shape

  for i in range(N):
    for j in range(M):
      if Q[i,j] == 255:
        mask_edge_manual[i,j,:] = [255,0,0]

  #Graficando contorno de mascara creada con superpixeles
  mask_edge_sp = mark_boundaries(img, segments_slic)[:,:,(2,1,0)]

  for i in range(N):
    for j in range(M):
      if S[i,j] == 255:
        mask_edge_sp[i,j,:] = [0,0,255]

  img_comparative = mask_edge_manual.copy()

  #Graficando ambas
  for i in range(N):
    for j in range(M):
      if S[i,j] == 255:
        img_comparative[i,j,:] = [0,0,255]


  return mask_sp, mask_edge_sp, mask_edge_manual, img_comparative

def encoder(PATH):
  X = []
  names = os.listdir(PATH)
  names.sort()
  for i in range(len(names)):
    img_path = "{}/{}".format(PATH,names[i])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X.append([img])
  X = np.array(X)
  X = np.squeeze(X)
  return X

def iou_basic(img_ref, img_sp):
  (N,M) = img_ref[:,:,0].shape
  inter = 0
  union_ref = 0
  union_sp = 0
  img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
  img_sp = cv2.cvtColor(img_sp, cv2.COLOR_BGR2GRAY)
  for r in range(N):
    for c in range(M):
        if img_ref[r,c] == 255: union_ref+=1
        if img_sp[r,c] == 255: union_sp+=1
        if (img_ref[r,c] == 255 and img_sp[r,c] == 255):
          inter+=1
  iou = inter/(union_ref+union_sp-inter)
  return iou

def calculate_iou(img_ref, img_sp):

  inter = 0
  img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
  img_sp = cv2.cvtColor(img_sp, cv2.COLOR_BGR2GRAY)

  img_ref = np.array(img_ref); img_sp = np.array(img_sp)
  u_ref = np.where(img_ref == 255); u_sp = np.where(img_sp == 255)
  num_pxl_ref = len(u_ref[1]); num_pxl_sp = len(u_sp[1])
  inter = np.shape(np.where(img_sp[u_ref]==255))[1]
  d = num_pxl_ref + num_pxl_sp - inter

  if d==0:iou = 0
  else: iou = inter/d
  return iou

def centroide(masks, segments_slic):
  #Centoride
  x_c = []; y_c = []
  for n in range(np.max(segments_slic)):
    cen = ndimage.center_of_mass(masks[n])
    x_c.append(int(cen[1]))
    y_c.append(int(cen[0]))
  return x_c, y_c

def create_df(name_c1,name_c2,data_c1,data_c2):
  df = pd.DataFrame()
  datos = {
      "{}".format(name_c1) : data_c1,
      "{}".format(name_c2): data_c2
  }
  df = pd.DataFrame(datos)
  return df

def save_pd(df,file ,id):
  path = "/content/drive/MyDrive/FootUlcerSegmentationChallenge/Segmentation_Estadistical/Evaluaciones/{}".format(file)
  df.to_csv("{}/prueba_{}.csv".format(path,id), header=True, index=False)

def create_df_feature_color(var_,med_,asy_,fec_,int_,label,names,cont_,x_c_,y_c_):
  #ingresar la imágen en formator RGB
  df = pd.DataFrame()
  datos = {
      'N_img': names,
      'Cantidad': cont_,
      'x_c': x_c_,
      'y_c': y_c_,
      'var_b': var_[2], 'var_g': var_[1], 'var_r': var_[0],
      'mean_b': med_[2], 'mean_g': med_[1], 'mean_r': med_[0],
      'F_b': fec_[2], 'F_g': fec_[1], 'F_r': fec_[0],
      'as_b': asy_[2], 'as_g': asy_[1], 'as_r': asy_[0],
      'I_b': int_[2], 'I_g': int_[1], 'I_r': int_[0],
      'Wound': label
  }

  df = pd.DataFrame(datos)
  return df

def contador_pxls(segments_slic):
  cont = []
  for i in range(np.max(segments_slic)+1):
    num = segments_slic[np.where(segments_slic==i)] #buscando pixeles cpn la misma etiqueta
    n = np.shape(num)[0] #contando pixeles con la misma etiqueta
    cont.append(n)
  return cont

def flatten(arr):
  new_arr = []
  for i in range(len(arr)):
    for j in range(len(arr[i])):
      new_arr.append(arr[i][j])
  return new_arr

def labels_visual(img, mascara, segments_slic, clas):
  marcadores = []
  for i in clas:
    marcadores.append(np.where(segments_slic==i))

  #creamos la máscara usando superpixeles
  mask_sp = np.zeros(img.shape, dtype="uint8")
  for idx in range(len(marcadores)):
    mask_sp[marcadores[idx]] = 255

  #contorno de mascara con sp
  S = cv2.Canny(mask_sp,250,255)

  #contorno de mascara
  Q = cv2.Canny(mascara,250,255)

  #Graficando contorno de mascara manual
  mask_edge_manual = mark_boundaries(img, segments_slic)[:,:,(2,1,0)]
  (N,M) = img[:,:,0].shape

  for i in range(N):
    for j in range(M):
      if Q[i,j] == 255:
        mask_edge_manual[i,j,:] = [255,0,0]

  #Graficando contorno de mascara creada con superpixeles
  mask_edge_sp = mark_boundaries(img, segments_slic)[:,:,(2,1,0)]

  for i in range(N):
    for j in range(M):
      if S[i,j] == 255:
        mask_edge_sp[i,j,:] = [0,0,255]

  img_comparative = mask_edge_manual.copy()

  #Graficando ambas
  for i in range(N):
    for j in range(M):
      if S[i,j] == 255:
        img_comparative[i,j,:] = [0,0,255]


  return mask_sp, mask_edge_sp, mask_edge_manual, img_comparative

def save_img(images,labels,path_out,wound):
    #visualizando
    r, c = 2, 2
    cont = 0
    fig = plt.figure(figsize=(5*c, 5*r))
    for _r in range(r):
        for _c in range(c):
            plt.subplot(r, c, _r*c + _c + 1)
            img = images[cont]
            plt.imshow(img[:,:,[2,1,0]])
            label = labels[cont]
            plt.title(label)
            plt.axis(False)
            cont+=1
    iou = calculate_iou(images[1], images[0])
    fig.suptitle("IoU = {:.2f} and N_spxl = {}".format(iou, len(wound)))
    plt.savefig("{}/Image.jpg".format(path_out))

def calculate_iou(img_ref, img_sp):

  inter = 0
  img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
  img_sp = cv2.cvtColor(img_sp, cv2.COLOR_BGR2GRAY)

  img_ref = np.array(img_ref); img_sp = np.array(img_sp)
  u_ref = np.where(img_ref == 255); u_sp = np.where(img_sp == 255)
  num_pxl_ref = len(u_ref[1]); num_pxl_sp = len(u_sp[1])
  inter = np.shape(np.where(img_sp[u_ref]==255))[1]
  d = num_pxl_ref + num_pxl_sp - inter

  if d==0:iou = 0
  else: iou = inter/d
  return iou