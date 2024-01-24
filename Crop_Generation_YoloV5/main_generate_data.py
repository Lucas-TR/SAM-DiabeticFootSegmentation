# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:03:42 2022

@author: Nicolas
"""
import extractor_features_SLIC as vector_slic


'''
* n_segmentes: The (approximate) number of labels in the segmented output image.
* compactness: Balances color proximity and space proximity. Higher values give more weight to space proximity, making superpixel shapes more square/cubic.
* sigma: Width of Gaussian smoothing kernel for pre-processing for each dimension of the image.
* threshold: Indicates the number of pixels that must intersect with the mask to be considered part of the lesion.
* path_img: packet where the masks are located.
* path_out: packet where the results will be stored.
* file_per_image: (True) to create a csv for each image.
'''

#ruta_images_val = "/content/drive/MyDrive/FootUlcerSegmentationChallenge/DataBase/VRI/Data_segmentation_DFUS/coor_txt_test/crops_test"
#ruta_labels_val = "/content/drive/MyDrive/FootUlcerSegmentationChallenge/DataBase/VRI/Data_segmentation_DFUS/coor_txt_test/crops_test_masks"
path_img = "Images"
path_mask = "Masks"
path_out = "."


n_segment = 100; compactness=10; sigma=1 ; threshold = 180; file_per_image = True

vector_slic.feature_slic(n_segment, compactness, sigma, threshold , path_img, path_mask, path_out,file_per_image)




