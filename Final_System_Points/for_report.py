# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 02:56:40 2022

@author: nicol
"""

import tool

PATH = 'code_informe'

Images = tool.encoder(PATH)

lista_names = tool.names_sort(PATH)
names = tool.split_dot(lista_names)

iou = []
for i in range(len(Images)):
    a = Images[i]
    b=  Images[3]
    iou.append(tool.calculate_iou(Images[3], Images[i]))
    

name = 'README.txt'
README = '{}/{}'.format(PATH,name)
with open (README, 'w') as f:
   f.write('IOU \n\n')
   for idx in range(len(iou)):
       str_iou = '{}'.format(iou[idx])
       f.write('{} : {} \n'.format(names[idx],str_iou))
   f.close()
print("Finish...!")