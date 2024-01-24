# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 23:15:25 2022

@author: Nicolas
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
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from keras.applications.vgg16 import VGG16
import generate_feature_map as gen_vgg16
import random
import extractor_features_rgb_and_vgg16 as vector_slic_rgb_vgg16
import extractor_features_vgg16 as vector_slic_vgg16
import extractor_features_rgb as vector_slic_rgb


'''
<-----TOOL General----->
'''
def names_sort(PATH):
    names = os.listdir(PATH) 
    names.sort() #ordenar
    return names

def split_dot(lista):
    before_dot = []
    for name in lista:
        list_unit = name.split('.')
        before_dot.append(list_unit[0])
    return before_dot

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

def new_file(PATH,file,i):
          files = os.listdir(PATH)
          if file in files and i==0: 
              shutil.rmtree('{}/{}'.format(PATH, file))
              os.mkdir('{}/{}'.format(PATH, file))
          else:
              if i==0:
                  os.mkdir('{}/{}'.format(PATH, file))

                  
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
  img_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
  img_sp = cv2.cvtColor(img_sp, cv2.COLOR_RGB2GRAY)

  img_ref = np.array(img_ref); img_sp = np.array(img_sp)
  u_ref = np.where(img_ref == 255); u_sp = np.where(img_sp == 255)
  num_pxl_ref = len(u_ref[1]); num_pxl_sp = len(u_sp[1])
  inter = np.shape(np.where(img_sp[u_ref]==255))[1]
  d = num_pxl_ref + num_pxl_sp - inter

  if d==0:iou = 0
  else: iou = inter/d
  return iou


                 
'''
<-----TOOL for SLIC----->
'''
        
def centroide(masks, segments_slic):
  #Centoride
  x_c = []; y_c = []
  for n in range(np.max(segments_slic)+1):
    cen = ndimage.center_of_mass(masks[n])
    x_c.append(int(cen[1]))
    y_c.append(int(cen[0]))
  return x_c, y_c


def create_img(img, mask_ref, segments_slic, clas, path_out, wound, name_id, predict=False, gc = False):
    if predict:
        mask_sp, mask_edge_sp, mask_edge_manual, img_comparative = labels_visual(img, mask_ref, segments_slic, clas)
        if gc:
            mask_sp = grub_cut(img, mask_sp) # la imagen es procesada por 
        
        images = [mask_sp, mask_ref, mask_edge_sp, mask_edge_manual]
        labels = ['mask SP', 'mask Manual', 'Edge SP', 'Edge Manual']
        iou = save_img(images,labels,path_out,wound,name_id,predict=True)
        return iou
    else:
        mask_sp, mask_edge_sp, mask_edge_manual, img_comparative = labels_visual(img, mask_ref, segments_slic, clas)
        
        images = [mask_sp, mask_ref, mask_edge_sp, mask_edge_manual]
        labels = ['mask SP', 'mask Manual', 'Edge SP', 'Edge Manual']
        save_img(images,labels,path_out,wound,name_id)

    
def create_masks(img,segments_slic):
  superPixels = []; masks = []
  for i in range(np.max(segments_slic)+1):
    superPixels.append(np.where(segments_slic==i))
    mask = np.zeros(img.shape, dtype="uint8")
    mask[superPixels[i]] = 255
    masks.append(mask[:,:,0])
  return masks, superPixels

#Extraer caracteristicas de color para una sola imagen
def feature_color( img, ch, masks, sup_pxls, segments_slic): 
  var_ch = []; me_ch = []; as_ch = []; f_ch = []; i_ch = []
  for i in range(np.max(segments_slic)+1):
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
  for i in range(np.max(segments_slic)+1):
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

def create_df_feature_color(var_,med_,asy_,fec_,int_,names,cont_,x_c_,y_c_,label='',bool_label=True):
  #ingresar la imágen en formator RGB
  df = pd.DataFrame()
  datos = {
      'N_img': names,
      'Cantidad': cont_,
      'x_c': x_c_,
      'y_c': y_c_
  }
  #"R-G-B"
  for i in range(len(var_)):
      datos[ "var_ch_{}".format(i+1) ] = var_[i]
  for i in range(len(var_)):
      datos[ "mean_ch_{}".format(i+1) ] = med_[i]
  for i in range(len(var_)):
      datos[ "F_ch_{}".format(i+1) ] = fec_[i]
  for i in range(len(var_)):
    datos[ "as_ch_{}".format(i+1) ] = asy_[i]
  for i in range(len(var_)):
      datos[ "i_ch_{}".format(i+1) ] = int_[i]
  if bool_label:
      datos[ "Wound" ] = label
  
  
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


def save_img(images,labels,path_out,wound,name_id, predict = False):
    #visualizando
    r, c = 2, 2
    cont = 0
    fig = plt.figure(figsize=(5*c, 5*r))
    for _r in range(r):
        for _c in range(c):
            plt.subplot(r, c, _r*c + _c + 1)
            img = images[cont]
            img= img[:,:,[2,1,0]]
            img_aux = img.astype(int)
          #  img_aux = img_aux*255
            plt.imshow(img)
            label = labels[cont]
            plt.title(label)
            plt.axis(False)
            cont+=1
    iou = calculate_iou(images[1], images[0])
    if predict:
        name_id_arr = name_id.split('_')
        title_id = name_id_arr[-1]
        fig.suptitle("{}\n\n IoU = {:.2f} and N_spxl = {}".format(title_id,iou, len(wound)))
        plt.savefig("{}/{}.jpg".format(path_out, name_id))
        plt.close()
        return iou
        
    else:
        fig.suptitle("IoU = {:.2f} and N_spxl = {}".format(iou, len(wound)))
        plt.savefig("{}/{}.jpg".format(path_out, name_id))
        plt.close()

def save_img_EDSR(PATH, img, name):
    file = '{}/{}.jpg'.format(PATH,name)
    cv2.imwrite(file,img)
    cv2.waitKey(0)
 
    
'''
<-----TOOL for traning----->
'''

def separate_data_train(df):
    # Loading data
    #df = pd.read_csv('{}'.format(name_feature_csv))
    #df.head()
     
    del(df['N_img'],df['Cantidad'])
    #df.head()

    """## Scaling data"""

    array = df.values
    # separate array into input and output components

    lim_in = 0
    lim_sup = array.shape[1]-1

    X = array[:,lim_in:lim_sup]

    #Rescale
    scaler = MaxAbsScaler().fit(X) #applicate (positive scale [0,1]) for now
    rescaledX = scaler.transform(X)

    #Create of dateframe
    df2 = pd.DataFrame()
    df2 = pd.DataFrame(rescaledX)

    #Addd one column of labels
    df2['Wound'] = df.Wound

    #Rename the columnos
    df2.columns = df.columns

    df2.head()

    #df = df[0:int(df.shape[0]*5/100)]
    #df2.shape

    """### **Analyzing data**"""

    # We visualize the distribution of the data
    '''
    fig = plt.figure(figsize = (20,20))
    ax = fig.gca()
    df2.hist(ax=ax)
    plt.show()

    # Correlatioin of the features

    correlation = df2.corr()
    plt.figure(figsize=(20,20))
    ax = sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='viridis')
    plt.title('Matriz de correlación')
    plt.show()
    '''
    """## Creando SVM"""

    #Train Data
    #X = df.drop(['Wound','N_img'], axis='columns')
    X = df2.drop(['Wound'], axis='columns')
    y = df2.Wound
    #X = df.drop(['Wound'], axis='columns')
    #y = df.Wound
    
    return df2



#############################################################
#Prediction
#############################################################
def detector_predict(wound, segments_slic):
  clas = [] # se guardara el ID del superpixel que contiene lesión
  #wound = [] # Se etiquetara al superpixel
  for i in range(np.max(segments_slic)+1):
    if wound[i]==1:
      clas.append(i)
  return clas

def separate_data_train(df):
    # Loading data
    #df.head()
     
    del(df['N_img'],df['Cantidad'])
    #df.head()

    """## Scaling data"""

    array = df.values
    # separate array into input and output components

    lim_in = 0
    lim_sup = array.shape[1]-1

    X = array[:,lim_in:lim_sup]

    #Rescale
    scaler = MaxAbsScaler().fit(X)
    rescaledX = scaler.transform(X)

    #Create of dateframe
    df2 = pd.DataFrame()
    df2 = pd.DataFrame(rescaledX)

    #Addd one column of labels
    df2['Wound'] = df.Wound

    #Rename the columnos
    df2.columns = df.columns

    #df2.head()

    #df = df[0:int(df.shape[0]*5/100)]
    #df2.shape

    """### **Analyzing data**"""

    # We visualize the distribution of the data
    '''
    fig = plt.figure(figsize = (20,20))
    ax = fig.gca()
    df2.hist(ax=ax)
    plt.show()

    # Correlatioin of the features

    correlation = df2.corr()
    plt.figure(figsize=(20,20))
    ax = sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='viridis')
    plt.title('Matriz de correlación')
    plt.show()
    '''
    """## Creando SVM"""

    #Train Data
    #X = df.drop(['Wound','N_img'], axis='columns')
    X = df2.drop(['Wound'], axis='columns')
    y = df2.Wound
    #X = df.drop(['Wound'], axis='columns')
    #y = df.Wound
    return X, y

def encoder_files(PATH):
  names = os.listdir(PATH) 
  return names

def split_dot(lista):
    before_dot = []
    for name in lista:
        list_unit = name.split('.')
        before_dot.append(list_unit[0])
    return before_dot


def train_SVM(path_out, name_training, file_train, test_n):
    models = []
    file_train = '{}/{}'.format(path_out,file_train)
    data_train =  encoder_files(file_train)
    data_whithout_dot = split_dot(data_train)
    file_save_data = 'Data_train'
    
    #temporal:
    obs = []
    
    predictions_arr =[]
    percentage_arr = []
    res_arr = []
    len_test = []
    for i in range(len(data_train)):
        new_file(path_out, name_training, i)
        new_file('{}/{}'.format(path_out,name_training), file_save_data, i)
        #path_save = "{}/{}/{}".format(path_out,name_training, 'Model_'+data_whithout_dot[i])    
        #os.mkdir(path_save)
        
        # Loading data
        filename = os.path.join(file_train ,data_train[i])
        df = pd.read_csv('{}'.format(filename), index_col=0)
        obs.append(df)
        X , y =  separate_data_train(df)
        
        if i==0:
            #Model Training
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= float(test_n))
            ##I have save theese files
            id_X_train = X_train.index
            id_y_train = y_train.index
            id_X_test = X_test.index
            id_y_test = y_test.index
            X_train.to_csv("{}/{}/{}/X_train.csv".format(path_out,name_training,file_save_data))
            X_test.to_csv("{}/{}/{}/X_test.csv".format(path_out,name_training,file_save_data))
            y_train.to_csv("{}/{}/{}/y_train.csv".format(path_out,name_training,file_save_data))
            y_test.to_csv("{}/{}/{}/y_test.csv".format(path_out,name_training,file_save_data))
        else:
            X_train = X.loc[ id_X_train]    
            y_train = y.loc[ id_y_train]    
            X_test = X.loc[ id_X_test]    
            y_test = y.loc[ id_y_test]    

                
        model = SVC(kernel='linear')
        
        #filtro
        X_train = X_train.fillna(0)
        
        model.fit(X_train, y_train)
        

        
        name = "SVM_model_{}".format(data_whithout_dot[i])
        #Save model
        joblib.dump(model, "{}/{}/{}.joblib".format(path_out,name_training,name)) #guarda el modelo
        models.append(model)
        
        #Performance
        X_test = X_test.fillna(0)
        predictions = model.predict(X_test)
        percentage = model.score(X_test, y_test)     
        res = confusion_matrix(y_test, predictions)
        
        #Save
        predictions_arr.append(predictions)
        percentage_arr.append(percentage*100)
        res_arr.append(res)
        len_test.append(len(X_test))
        
    name = 'README.txt'
    README = '{}/{}/{}'.format(path_out,name_training,name)        
    with open (README, 'w') as f:
        iden =0
        for name_model in data_whithout_dot:
            f.write('Training_model: {}\n'.format(name_model))
            f.write("Confusion Matrix: \n{}\n".format(res_arr[iden]))
            f.write('Test Set: {} \n'.format(len_test[iden]))
            f.write('Accuracy = {:.2f} %\n\n\n'.format(percentage_arr[iden]))
            iden = iden + 1
        f.close()          
        
    return models, obs


'''
    #Performance
    predictions = model.predict(X_test)
    print(predictions)
    percentage = model.score(X_test, y_test)
    res = confusion_matrix(y_test, predictions)
    print("Confusion Matrix")
    print(res)
    print(f"Test Set: {len(X_test)}")
    print(f"Accuracy = {percentage*100} %")
    
    #almacenando y cargando el modelo
    

    
    #Load model
    #model = joblib.load("{}/Muestra_2.joblib".format(PATH)) #carga el modelo
'''


def prediction_SVM(path_out_origin, path_out, path_img, path_mask, name_prediction, file_models,n_segment, compactness, sigma, threshold , layer, name_image):
    file_visual_prediction = 'Vista'
    file_visual_grabcut = 'Vista_grabcut'
    PATH = '{}/{}'.format(path_out_origin,file_models)
    data_models = encoder_files(PATH)
    data_models.remove('Data_train')
    data_models.remove('README.txt')
    name_models = split_dot(data_models)
    img = encoder(path_img)
    mask = encoder(path_mask)
    #new_file(path_out, name_prediction, 0)
    
    file_per_image = True; file_iou_image = True
    
    #PATH = '{}/{}'.format(path_out,name_prediction)
    PATH = path_out
    
    iou_s = []
    iou_g = []
    for i in range(len(name_models)):
        arr_id = name_models[i].split('_')
        idx_end = arr_id[-1]
        if(idx_end =='RGB'):
            name_experiment = "Results_RGB"
            df = vector_slic_rgb.feature_slic(name_experiment, n_segment, compactness, sigma, threshold, path_img, path_mask, PATH,file_per_image,file_iou_image)
        if(idx_end =='VGG'):
            name_experiment = "Results_VGG16"
            df = vector_slic_vgg16.feature_slic(name_experiment, n_segment, compactness, sigma, threshold , layer, path_img, path_mask, PATH,file_per_image,file_iou_image)
        if(idx_end =='RGB-VGG'):
            name_experiment = "Results_RGB_VGG16"
            df = vector_slic_rgb_vgg16.feature_slic(name_experiment, n_segment, compactness, sigma, threshold , layer, path_img, path_mask, PATH,file_per_image,file_iou_image)
    
        
        X , y = separate_data_train(df)
            
        #Load model
        model = joblib.load("{}/{}/{}".format(path_out_origin,file_models,data_models[i])) 
        
        X = X.fillna(0)
        predictions = model.predict(X)
        
        
        #generate_segments_slic
        segments_slic = slic(img, n_segment , compactness=compactness , sigma = sigma)
        if segments_slic[0][0] != 0:
            segments_slic = segments_slic - 1 #segments_slic has to start at 0
        
        
        cont_pxl = contador_pxls(segments_slic)
        masks, positions_pxl = create_masks(img, segments_slic)
        
        #la imagen debe estar en formato RGB
        clas = detector_predict(predictions, segments_slic)
        
        mask_sp, mask_edge_sp, mask_edge_manual, img_comparative = labels_visual(img, mask, segments_slic, predictions)
        
        Path_model = '{}/{}/{}'.format(path_out_origin,'Predictions',name_prediction)
        new_file(Path_model, file_visual_prediction, i)
        
        iou = create_img(img, mask, segments_slic, clas, '{}/{}'.format(Path_model,file_visual_prediction), predictions, name_models[i],predict = True)
        
        iou_s.append(iou)
        
        
        Path_model = '{}/{}/{}'.format(path_out_origin,'Predictions',name_prediction)
        new_file(Path_model, file_visual_grabcut, i)
        
        
        iou = create_img(img, mask, segments_slic, clas, '{}/{}'.format(Path_model,file_visual_grabcut), predictions, name_models[i],predict = True, gc=True)
        
        iou_g.append(iou)
    
    return iou_s, iou_g
        


####
def view_clas_wound(df_aux,clas):
    df = df_aux.copy()
    for i in range(len(df[clas])):
        if df[clas][i] == 0:
            df[clas][i] = 'no wound'
        else:
            df[clas][i] = 'wound'
    sns.set_style("whitegrid")
    sns.countplot(x=clas, data = df, hue=clas)
    
    return df

def predictions(path_out_origin, path_out, path_img_test, path_mask_test, file_models, n_segment, compactness, sigma, threshold , layer):
    file_visual_prediction = 'Predictions'
    new_file(path_out, file_visual_prediction, 0)
    PATH = '{}/{}'.format(path_out_origin, path_img_test)
    PATH_mask = '{}/{}'.format(path_out_origin, path_mask_test)
    data_images = encoder_files(PATH)
    name_images = split_dot(data_images)

    #img = encoder(path_img_test)
    #mask = encoder(path_mask_test)
   
    iou_rgb = []; iou_rgb_grap = []
    iou_vgg = []; iou_vgg_grap = []
    iou_rgb_vgg = []; iou_rgb_vgg_grap = []
   
    for i in range(len(name_images)):
        #new_file(PATH, file, 0)
        name_prediction = '{}_{}'.format("Prediction", name_images[i])
        new_path_out ='{}/{}'.format(path_out,"Predictions") 
        new_file(new_path_out,name_prediction,0)
        
        new_file('{}/{}'.format(new_path_out,name_prediction),'image',0)
        shutil.copy('{}/{}'.format(PATH,data_images[i]), '{}/{}/{}/{}'.format(new_path_out,name_prediction,'image',data_images[i]))
        path_img = '{}/{}/{}'.format(new_path_out, name_prediction, 'image')
        
        new_file('{}/{}/{}'.format(path_out,"Predictions",name_prediction),'mask',0)
        shutil.copy('{}/{}'.format(PATH_mask ,data_images[i]), '{}/{}/{}/{}'.format(new_path_out,name_prediction,'mask',data_images[i]))
        path_mask = '{}/{}/{}'.format(new_path_out, name_prediction, 'mask')
        
        path_origin_aux = path_out
        path_out_2 = '{}/{}'.format(new_path_out,name_prediction)
        
        list_iou , list_iou_g = prediction_SVM(path_origin_aux, path_out_2, path_img, path_mask, name_prediction, file_models,n_segment, compactness, sigma, threshold , layer,name_images[i])
        
        #el orden importa
        iou_rgb.append(list_iou[0]); iou_rgb_grap.append(list_iou_g[0])
        iou_rgb_vgg.append(list_iou[1]); iou_rgb_vgg_grap.append(list_iou_g[1])
        iou_vgg.append(list_iou[2]); iou_vgg_grap.append(list_iou_g[2])
    
    iou_prom = [sum(iou_rgb)/len(iou_rgb), sum(iou_rgb_grap)/len(iou_rgb_grap) , sum(iou_vgg)/len(iou_vgg), sum(iou_vgg_grap)/len(iou_vgg_grap), sum(iou_rgb_vgg)/len(iou_rgb_vgg), sum(iou_rgb_vgg_grap)/len(iou_rgb_vgg_grap)]
    name = 'README.txt'
    README = '{}/{}/{}'.format(path_out, 'Predictions',name) 
    with open (README, 'w') as f:
        f.write('Predictions:\n\n')
        f.write('number of test images: {}'.format(len(name_images)))
        f.write('\nIOU_prom_rgb: {:.2f} || IOU_prom_rgb_grab: {:.2f}\n IOU_prom_vgg: {:.2f} || IOU_prom_vgg_grab: {:.2f}  \n IOU_prom_rgb_vgg: {:.2f} || IOU_prom_rgb_vgg_grab: {:.2f} \n'.format(iou_prom[0],iou_prom[1],iou_prom[2],iou_prom[3],iou_prom[4],iou_prom[5]))
        for i in range(len(iou_rgb)):          
            f.write("\n {}:\niou_rgb: {:.2f} || iou_rgb_grab: {:.2f}\n iou_vgg: {:.2f} || iou_vgg_grab: {:.2f}\n iou_rgb_vgg: {:.2f} || iou_rgb_vgg_grab: {:.2f}".format(name_images[i], iou_rgb[i], iou_rgb_grap[i], iou_vgg[i], iou_vgg_grap[i], iou_rgb_vgg[i], iou_rgb_vgg_grap[i]))
        f.close()
        
        
        
def grub_cut(img, mask):


    newmask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) #pasar a escala de gris para que solo sea una matriz, pero no esta normalizado


    ret, orig_mask = cv2.threshold(newmask, 20, 255, cv2.THRESH_BINARY) #normalizanos, es decir los pixeles solo tomaran 0 o 255
    orig_mask = orig_mask/255 #los pixeles estaran entre 0 y 1

    orig_mask = np.array(orig_mask, dtype=np.uint8)

    #esto es fijo
    bgdModel = np.zeros((1,65),dtype = np.float64)
    fgdModel = np.zeros((1,65),dtype = np.float64)


    mask_grab, bgdModel, fgdModel = cv2.grabCut(img, orig_mask , None,bgdModel,fgdModel,50,cv2.GC_INIT_WITH_MASK)


    mask_grab = cv2.cvtColor(mask_grab, cv2.COLOR_GRAY2BGR)
    mask_grab = mask_grab *255
    
    return mask_grab

    '''
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask[:,:,np.newaxis]
    plt.imshow(img[:,:,(2,1,0)]),plt.colorbar(),plt.show()
    '''
def save_img_unet(images,labels,path_out,name_id, predict = False):
    #visualizando
    r, c = 2, 2
    cont = 0
    fig = plt.figure(figsize=(5*c, 5*r))
    for _r in range(r):
        for _c in range(c):
            plt.subplot(r, c, _r*c + _c + 1)
            img = images[cont]
            img= img[:,:,[2,1,0]]
            img_aux = img.astype(int)
          #  img_aux = img_aux*255
            plt.imshow(img)
            label = labels[cont]
            plt.title(label)
            plt.axis(False)
            cont+=1
    iou = calculate_iou(images[1], images[0])
    if predict:
        name_id_arr = name_id.split('_')
        title_id = name_id_arr[-1]
        fig.suptitle("{}\n\n IoU = {:.2f}".format(title_id,iou))
        plt.savefig("{}/{}.jpg".format(path_out, name_id))
        plt.close()
        return iou
        
    else:
        fig.suptitle("IoU = {:.2f}".format(iou))
        plt.savefig("{}/{}.jpg".format(path_out, name_id))
        plt.close()
        
def labels_visual_unet(img, mask, mask_sp):

  img_unet = img.copy()

  #contorno de mascara con sp
  S = cv2.Canny(mask_sp,250,255)

  #contorno de mascara
  Q = cv2.Canny(mask,250,255)

  #Graficando contorno de mascara manual
  (N,M) = img[:,:,0].shape

  for i in range(N):
    for j in range(M):
      if Q[i,j] == 255:
        img[i,j,:] = [255,0,0]



  for i in range(N):
    for j in range(M):
      if S[i,j] == 255:
        img_unet[i,j,:] = [0,0,255]

  img_comparative = img.copy()

  #Graficando ambas
  for i in range(N):
    for j in range(M):
      if S[i,j] == 255:
        img_comparative[i,j,:] = [0,0,255]
        
  
  return mask, mask_sp, img, img_comparative


def draw_bbox(img, bboxes, color=(255, 0, 0), thickness=2):
    """
    Dibuja bounding boxes en la imagen.

    Args:
    img: Imagen en la que dibujar los bounding boxes.
    bboxes: Lista de tuplas de bounding boxes, donde cada tupla contiene una lista de las coordenadas
            del bounding box y el valor entero de la clase correspondiente.
    color: Color del bounding box en formato BGR. Por defecto es rojo.
    thickness: Grosor de las líneas del bounding box. Por defecto es 2.

    Returns:
    img: Imagen con los bounding boxes dibujados.
    """
    for bbox in bboxes:
        coordinates, _ = bbox
        start_point = (int(coordinates[0]), int(coordinates[1]))  # Coordenadas de la esquina superior izquierda
        end_point = (int(coordinates[2]), int(coordinates[3]))  # Coordenadas de la esquina inferior derecha
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
    
    return img

def segment_image(image, input_box, predictor):
    """
    Genera la segmentación de una imagen utilizando un modelo SAM.

    Args:
    input_box: Bounding box de entrada en formato numpy array [x1, y1, x2, y2].
    predictor: Modelo SAM preentrenado.

    Returns:
    masks: Máscara de segmentación resultante.
    """
    predictor.set_image(image)

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    return masks


def unir_mascaras(mascaras):
    mascara_unida = np.zeros_like(mascaras[0], dtype=bool)
    
    for mascara in mascaras:
        mascara_unida = np.logical_or(mascara_unida, mascara)
    
    return mascara_unida

def create_new_file(ruta_carpeta):
    # Verificar si la carpeta ya existe
    if not os.path.exists(ruta_carpeta):
        # Crear la carpeta
        os.makedirs(ruta_carpeta)
        print("Carpeta creada exitosamente.")
    else:
        print("La carpeta ya existe.")