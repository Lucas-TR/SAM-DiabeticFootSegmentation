# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:24:14 2022

@author: Nicolas
"""


from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.transform import resize



def features_maps_VGG16(model, img,idn, layer,path=''):

    # redefine model to output right after the first hidden layer
    model = Model(inputs=model.inputs, outputs=model.layers[layer].output)
    #model.summary()
    # load the image with the required shape
    
    height, width , ch = img.shape
    img = cv2.resize(img, (224,224))
    
    #plt.imshow(img)
    #img = load_img('wound.jpg', target_size=(224, 224))
    # convert the image to an array
    #img = img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img = expand_dims(img, axis=0)
    # prepare the image (e.g. scale pixel values for the vgg)
    img = preprocess_input(img)
    # get feature map for first hidden layer
    feature_maps = model.predict(img)
    feature_maps = feature_maps[0, :, :, :]
    feature_maps = resize(feature_maps, (height, width,feature_maps.shape[2]))
    #pack_imgs_VGG16(feature_maps, path,idn)
    
    return feature_maps

def pack_imgs_VGG16(feature_maps, path,idn):
    plt.close()
    r, c = 8, 8
    cont = 1
    fig = plt.figure(figsize=(3*c, 3*r))
    for _r in range(r):
        for _c in range(c):
            plt.subplot(r, c, _r*c + _c + 1)
            #img = feature_maps[0, :, :, cont-1]
            img = feature_maps[:,:,cont-1]
            plt.imshow(img)
            #plt.title(label)
            plt.axis(False)
            cont+=1
    plt.savefig("{}/{}.jpg".format(path,idn))
    plt.close()

