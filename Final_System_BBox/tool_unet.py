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

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC




'''
<-----TOOL General----->
'''

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


def create_img(img, mask_ref, segments_slic, clas,path_out,wound,name_id,predict=False, gc = False):
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

def create_df_feature_color(var_,med_,asy_,fec_,int_,label,names,cont_,x_c_,y_c_):
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


        
        
def grub_cut(img, mask):


    newmask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) #pasar a escala de gris para que solo sea una matriz, pero no esta normalizado


    ret, orig_mask = cv2.threshold(newmask, 20, 255, cv2.THRESH_BINARY) #normalizanos, es decir los pixeles solo tomaran 0 o 255
    orig_mask = orig_mask/255 #los pixeles estaran entre 0 y 1

    orig_mask = np.array(orig_mask, dtype=np.uint8)

    #esto es fijo
    bgdModel = np.zeros((1,65),dtype = np.float64)
    fgdModel = np.zeros((1,65),dtype = np.float64)


    mask_grab, bgdModel, fgdModel = cv2.grabCut(img, orig_mask , None,bgdModel,fgdModel,100,cv2.GC_INIT_WITH_MASK)


    mask_grab = cv2.cvtColor(mask_grab, cv2.COLOR_GRAY2BGR)
    mask_grab = mask_grab *255
    
    return mask_grab

    '''
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask[:,:,np.newaxis]
    plt.imshow(img[:,:,(2,1,0)]),plt.colorbar(),plt.show()
    '''

def labels_visual_unet(img, mascara, mask_sp):

  #contorno de mascara con sp
  S = cv2.Canny(mask_sp,250,255)

  #contorno de mascara
  Q = cv2.Canny(mascara,250,255)

  #Graficando contorno de mascara manual
  (N,M) = img[:,:,0].shape
  
  img_manual = img.copy()
  img_pred = img.copy()

  for i in range(N):
    for j in range(M):
      if Q[i,j] == 255:
        img_manual[i,j,:] = [255,0,0]

  #Graficando contorno de mascara creada con superpixeles
  for i in range(N):
    for j in range(M):
      if S[i,j] == 255:
        img_pred[i,j,:] = [0,0,255]

  img_comparative = img_manual.copy()

  #Graficando ambas
  for i in range(N):
    for j in range(M):
      if S[i,j] == 255:
        img_comparative[i,j,:] = [0,0,255]
        
  
  return mask_sp, img_pred, img_manual, img_comparative



def create_img_unet(img, mask_ref, mask_sp, path_out, name_id):
    
    mask_sp, img_pred, img_manual, img_comparative = labels_visual_unet(img, mask_ref, mask_sp)
        
    images = [mask_sp, mask_ref, img_pred, img_manual]
    labels = ['mask Unet', 'mask Manual', 'Edge Unet', 'Edge Manual']
    iou = save_img_unet(images, labels, path_out, name_id)


def save_img_unet(images,labels,path_out,name_id):
    #visualizando
    r, c = 2, 2
    cont = 0
    fig = plt.figure(figsize=(5*c, 5*r))
    for _r in range(r):
        for _c in range(c):
            plt.subplot(r, c, _r*c + _c + 1)
            img = images[cont]
            #img= img[:,:,[2,1,0]]
            img_aux = img.astype(int)
          #  img_aux = img_aux*255
            plt.imshow(img)
            label = labels[cont]
            plt.title(label)
            plt.axis(False)
            cont+=1
    iou = calculate_iou(images[1], images[0])

    name_id_arr = name_id.split('_')
    title_id = name_id_arr[-1]
    fig.suptitle("{}\n\n IoU = {:.2f} ".format(title_id,iou))
    plt.savefig("{}/{}.jpg".format(path_out, name_id))
    plt.close()
        