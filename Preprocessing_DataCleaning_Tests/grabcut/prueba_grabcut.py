# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 19:28:38 2022

@author: Nicolas
"""

import numpy as np
import cv2 
from matplotlib import pyplot as plt

img = cv2.imread('car.jpg')
plt.imshow(img[:,:,(2,1,0)])
plt.show()


newmask3 = cv2.imread('car_mask.jpg' )
newmask = cv2.imread('car_mask.jpg' , cv2.IMREAD_GRAYSCALE)


if newmask.all() == newmask2.all():
    print('son iguales')
else:
    print('son diferentes')


newmask2 = cv2.cvtColor(newmask3, cv2.COLOR_BGR2GRAY) #pasar a escala de gris para que solo sea una matriz, pero no esta normalizado


ret, orig_mask = cv2.threshold(newmask, 60, 255, cv2.THRESH_BINARY) #normalizanos, es decir los pixeles solo tomaran 0 o 255


newmask[orig_mask == 0] = 0
newmask[orig_mask == 255] = 1

orig_mask = orig_mask/255 #los pixeles estaran entre 0 y 1

orig_mask = np.array(orig_mask, dtype=np.uint8)

#esto es fijo
bgdModel = np.zeros((1,65),dtype = np.float64)
fgdModel = np.zeros((1,65),dtype = np.float64)


mask, bgdModel, fgdModel = cv2.grabCut(img, newmask , None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)


#mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#plt.imshow(mask*255)


mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
ver = mask[:,:,np.newaxis]
img = img*mask[:,:,np.newaxis]
plt.imshow(img[:,:,(2,1,0)]),plt.colorbar(),plt.show()

if mask.all() == ver.all():
    print('son iguales')
else:
    print('son diferentes')



import numpy as np
import cv2
from matplotlib import pyplot as plt
 
img = cv2.imread('0011_0.jpg')
mask = np.zeros(img.shape[:2],np.uint8)
 
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
 

rect = (200,50,1000,600)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_RECT)

 
mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask[:,:,np.newaxis]
 
plt.imshow(img),plt.colorbar(),plt.show()

plt.imshow(mask)


import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('0011_0.jpg')
newmask = cv2.imread('0011_0_mask.jpg',0)
plt.imshow(newmask)
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
# donde sea que esté marcado en blanco (primer plano seguro), cambiar mask=1
# donde sea que esté marcado en negro (fondo seguro), cambiar mask=0
mask[newmask == 0] = 2
mask[newmask == 255] = 3
 
#plt.imshow(mask)

cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

 
mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()

if mask.all() == ver.all():
    print('son iguales')
else:
    print('son diferentes')


'''
# organize imports
import numpy as np
import cv2
from matplotlib import pyplot as plt
  
# path to input image specified and
# image is loaded with imread command
image = cv2.imread('Qualityplus_1137_0.jpg')

# create a simple mask image similar
# to the loaded image, with the
# shape and return type
mask = np.zeros(image.shape[:2], np.uint8)
  
# specify the background and foreground model
# using numpy the array is constructed of 1 row
# and 65 columns, and all array elements are 0
# Data type for the array is np.float64 (default)
backgroundModel = np.zeros((1, 65), np.float64)
foregroundModel = np.zeros((1, 65), np.float64)
  
# define the Region of Interest (ROI)
# as the coordinates of the rectangle
# where the values are entered as
# (startingPoint_x, startingPoint_y, width, height)
# these coordinates are according to the input image
# it may vary for different images
rectangle = (20, 100, 150, 150)
  
# apply the grabcut algorithm with appropriate
# values as parameters, number of iterations = 3
# cv2.GC_INIT_WITH_RECT is used because
# of the rectangle mode is used
cv2.grabCut(image, mask, rectangle, 
            backgroundModel, foregroundModel,
            3, cv2.GC_INIT_WITH_RECT)
  
# In the new mask image, pixels will
# be marked with four flags
# four flags denote the background / foreground
# mask is changed, all the 0 and 2 pixels
# are converted to the background
# mask is changed, all the 1 and 3 pixels
# are now the part of the foreground
# the return type is also mentioned,
# this gives us the final mask
mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
  
# The final mask is multiplied with
# the input image to give the segmented image.
image = image * mask2[:, :, np.newaxis]
  
# output segmented image with colorbar
plt.imshow(image)
plt.colorbar()
plt.show()
'''