# -*- coding: utf-8 -*-
"""BIG_Segment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1khyYP2fUxs5qvNgNH1qsxZC7mIy6UMB3
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import zipfile
from random import randint
from PIL import Image
import re
from sklearn.model_selection import train_test_split
import skimage
import pandas as pd

import mahotas.features.texture as mht
import mahotas.features
import Go2BlackWhite
import Go2Mahotas

def Segmenta(img):
  Size=1200 # tamanho da foto
  Sub_Size=int(Size/5) # tamanho do fracionamento
  Row_Crop=1/2 # posicao do corte
  Crop=int(Size*Row_Crop)
  Num=50
  a=0
  b=1200
  c=100
  d=200

  ww=[]
  label=[]
  SizeWidth=[]

  for i in range(Num):

    x=randint(a, b)
    y=randint(a, b)
    Width=randint(c, d)
    img_1st=np.zeros((Width,Width)).astype(np.int64) 
    
    # estava for i in range(Width):

    for k in range(Width):
      for j in range(Width):
        
        size_x=Width+x
        size_y=Width+y
        
        if(size_x>=Size):
          x=Size-Width

        if(size_y>= Size):
          y=Size-Width

        img_1st[i,j]=np.copy(img[i+y,j+x])  

    ww.append(img_1st)
    SizeWidth.append(Width)
    nome = "W=" + str(Width)+" x="+str(x)+" y="+str(y)
    label.append(nome)

  Num=12
  Size=28
  img28_all=[]

  for i in range(Num):
    data=np.array(ww[i])
    img = Image.fromarray(data.astype('uint8'), mode='L')
    img=np.float32(img)
    img28=cv2.resize(img,(Size,Size), interpolation = cv2.INTER_AREA)
    img28_all.append(img28)

  img28_all=np.array(img28_all)

  Types_top=[]
  plt.figure(figsize=(15,15))

  for i in range(Num):
    Valor='Z'
    Types_top.append(Valor)

  return(ww)