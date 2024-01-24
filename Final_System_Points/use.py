# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 12:00:08 2022

@author: Nicolas
"""

import generate_EDSR as gen_EDSR
 
ruta_img = 'images' # path to the masks obtained with VGG annotator (csv)
ruta_masks = 'masks' # Path where the images are located

PATH = '.' 


# Running function

gen_EDSR.SR_EDSR(ruta_masks, ruta_img, PATH)

