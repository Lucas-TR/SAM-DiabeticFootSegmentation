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
  names.sort() #ordenar
  D = np.array(names)
  ind = np.where(D=='desktop.ini')
  if (ind[0] != 0):
      names.remove('desktop.ini')
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


def save_img(PATH, img, name):
    file = '{}/{}.jpg'.format(PATH,name)
    cv2.imwrite(file,img)
    cv2.waitKey(0)
 
    
 