# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 12:05:17 2022

@author: Nicolas
"""
import cv2
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import  slic, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float


name = '1014_3.jpg'

img = cv2.imread(name)

image_origin = cv2.imread(name)

sr = cv2.dnn_superres.DnnSuperResImpl_create()
 
path = "EDSR_x4.pb"
 
sr.readModel(path)
 
sr.setModel("edsr",4)
 
result = sr.upsample(img)
 
# Resized image
resized = cv2.resize(img,dsize=None,fx=4,fy=4)
 
plt.figure(figsize=(12,8))
plt.subplot(1,3,1)
# Original image
plt.imshow(img[:,:,::-1])
plt.subplot(1,3,2)
# SR upscaled
plt.imshow(result[:,:,::-1])
plt.subplot(1,3,3)
# OpenCV upscaled
plt.imshow(resized[:,:,::-1])
plt.show()


#imge_scale_mask = result

imge_scale = result


dst = cv2.addWeighted(imge_scale,0.5,imge_scale_mask,0.7,0)


plt.imshow(dst)
plt.show()





name = '1014_3.jpg'

img1 = cv2.imread(name)
name = '1014_3_mask.jpg'

img2 = cv2.imread(name)
dst2 = cv2.addWeighted(img1,0.5,img2,0.7,0)
plt.imshow(dst2)
plt.show()





#img = img_as_float(astronaut()[::2, ::2])
img_mask = result[:,:,::-1]
#call SLIC, with segment compactness = 10
segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
gradient = sobel(rgb2gray(img))
#call watershed with segment compactness = 0.1
segments_watershed = slic(image_origin, n_segments=250, compactness=10, sigma=1)
#count the number of segments
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
print('watershed number of segments: {}'.format(len(np.unique(segments_watershed))))

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

#ax[0, 0].imshow(mark_boundaries(img, segments_slic))
#ax[0, 0].set_title('SLIC with SR')

ax[0, 1].imshow(mark_boundaries(image_origin, segments_watershed))
ax[0, 1].set_title('SLIC without SR')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()