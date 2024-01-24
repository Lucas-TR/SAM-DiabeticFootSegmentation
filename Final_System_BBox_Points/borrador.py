# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 23:35:48 2022

@author: nicol
"""

import matplotlib.pyplot as plt
import cv2


img = cv2.imread('img.jpg')
mask = cv2.imread('mask.jpg')


plt.imshow(masked)