# -*- coding: utf-8 -*-
"""02_BIG_Segmentacao_e_salva_TODAS_calcula_entrada_FotoS_AGO_05_2020.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/ucfilho/marquesgabi_August_2020/blob/master/02_BIG_Segmentacao_e_salva_TODAS_calcula_entrada_FotoS_AGO_05_2020.ipynb
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import zipfile
#import random
from random import randint
from PIL import Image
import re
from sklearn.model_selection import train_test_split
#import scikit-image
import skimage
import pandas as pd

!pip install mahotas

import mahotas.features.texture as mht
import mahotas.features

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/ucfilho/marquesgabi_fev_2020 #clonar do Github
# %cd marquesgabi_fev_2020

import Go2BlackWhite
import Go2Mahotas

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/marquesgabi/Doutorado
# %cd Doutorado

Transfere='Fotos_Grandes_3cdAmostra.zip'
file_name = zipfile.ZipFile(Transfere, 'r')
file_name.extractall()

# Segmentation: start here......

# start top

#Start to use the big image
Size=1200 # tamanho da foto
Sub_Size=int(Size/5) # tamanho do fracionamento
Row_Crop=1/2 # posicao do corte
Crop=int(Size*Row_Crop)

ww,img_name=Go2BlackWhite.BlackWhite(Transfere,Size) #Pegamos a primeira foto Grande
img=ww[0]

print(img.shape)

plt.imshow(img, cmap = "gray")

Num=50

#First top

a=0
b=1200
c=100
d=200

ww=[]
label=[]
SizeWidth=[]  

for i in range(Num):
  #x=random.randint(a, b)
  #y=random.randint(a, b)
  #Width=random.randint(c, d)
  x=randint(a, b)
  y=randint(a, b)
  Width=randint(c, d)
  img_1st=np.zeros((Width,Width)).astype(np.int64)

  for i in range(Width):
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

#2nd top

Size=28
img28_all=[]
for i in range(Num):
  data=np.array(ww[i])
  img = Image.fromarray(data.astype('uint8'), mode='L')
  img=np.float32(img)
  img28=cv2.resize(img,(Size,Size), interpolation = cv2.INTER_AREA)
  img28_all.append(img28)

img28_all=np.array(img28_all)
print(img28_all.shape)

#3th top

plt.figure(figsize=(15,15))
for i in range(Num):
  plt.subplot(10,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(ww[i], cmap = "gray")
  plt.xlabel(i)

plt.subplots_adjust(bottom=0.15, right=0.8, top=2,hspace=0.1, wspace=0.1)

#4th top

Types_top=[]
plt.figure(figsize=(15,15))
for i in range(Num):
  #print('i=',i,'=====')
  #Valor=input('Tipo=')
  Valor='Z'
  Types_top.append(Valor)



# 5th top

img28_ravel_all=[]
for i in range(Num):
  img28_ravel=np.copy(img28_all[i].ravel())
  img28_ravel_all.append(img28_ravel)
  # img28_ravel_all.append(img28_all[i].ravel())

# 6th top

img28_top=pd.DataFrame(img28_ravel_all)
img28_top.insert(0,"Type",Types_top)
img28_top.insert(0, "Width", SizeWidth)

# Start middle

#Start to use the big image
Size=1200 # tamanho da foto
Sub_Size=int(Size/5) # tamanho do fracionamento
Row_Crop=1/2 # posicao do corte
Crop=int(Size*Row_Crop)
ww,img_name=Go2BlackWhite.BlackWhite(Transfere,Size) #Pegamos a primeira foto Grande
img=ww[0]

#First middle

a=0
b=1200
c=100
d=200

ww=[]
label=[]
SizeWidth=[]  
for i in range(Num):
  #x=random.randint(a, b)
  #y=random.randint(a, b)
  #Width=random.randint(c, d)
  x=randint(a, b)
  y=randint(a, b)
  Width=randint(c, d)
  img_1st=np.zeros((Width,Width)).astype(np.int64)

  for i in range(Width):
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

print([i+y,j+x])
print([x,y])
print([Size,Width])

print(np.array(img).shape)

#2nd middle

Size=28
img28_all=[]
for i in range(Num):
  data=np.array(ww[i])
  img = Image.fromarray(data.astype('uint8'), mode='L')
  img=np.float32(img)
  img28=cv2.resize(img,(Size,Size), interpolation = cv2.INTER_AREA)
  img28_all.append(img28)

img28_all=np.array(img28_all)
print(img28_all.shape)

#3th middle

plt.figure(figsize=(15,15))
for i in range(Num):
  plt.subplot(10,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(ww[i], cmap = "gray")
  plt.xlabel(i)

plt.subplots_adjust(bottom=0.15, right=0.8, top=2,hspace=0.1, wspace=0.1)

#4th middle

Types_middle=[]
plt.figure(figsize=(15,15))
for i in range(Num):
  #print('i=',i,'=====')
  Valor='Z'
  Types_middle.append(Valor)

# 5th middle

img28_ravel_all=[]
for i in range(Num):
  img28_ravel=np.copy(img28_all[i].ravel())
  img28_ravel_all.append(img28_ravel)
  # img28_ravel_all.append(img28_all[i].ravel())

# 6th middle

img28_middle=pd.DataFrame(img28_ravel_all)
img28_middle.insert(0,"Type",Types_middle)
img28_middle.insert(0, "Width", SizeWidth)

# start bottom

#Start to use the big image
Size=1200 # tamanho da foto
Sub_Size=int(Size/5) # tamanho do fracionamento
Row_Crop=1/2 # posicao do corte
Crop=int(Size*Row_Crop)
ww,img_name=Go2BlackWhite.BlackWhite(Transfere,Size) #Pegamos a primeira foto Grande
img=ww[0]

# First bottom

a=0
b=1200
c=100
d=200

ww=[]
label=[]
SizeWidth=[]

for i in range(Num):
  #x=random.randint(a, b)
  #y=random.randint(a, b)
  #Width=random.randint(c, d)
  x=randint(a, b)
  y=randint(a, b)
  Width=randint(c, d)
  img_1st=np.zeros((Width,Width)).astype(np.int64)

  for i in range(Width):
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

# 2nd bottom

Size=28
img28_all=[]
for i in range(Num):
  data=np.array(ww[i])
  img = Image.fromarray(data.astype('uint8'), mode='L')
  img=np.float32(img)
  img28=cv2.resize(img,(Size,Size), interpolation = cv2.INTER_AREA)
  img28_all.append(img28)

img28_all=np.array(img28_all)
print(img28_all.shape)

# 3th bottom

plt.figure(figsize=(15,15))
for i in range(Num):
  plt.subplot(10,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(ww[i], cmap = "gray")
  plt.xlabel(i)
  

plt.subplots_adjust(bottom=0.15, right=0.8, top=2,hspace=0.1, wspace=0.1)

# 4th bottom

Types_bottom=[]
plt.figure(figsize=(15,15))
for i in range(Num):
  # print('i=',i,'=====')
  Valor='Z'
  Types_bottom.append(Valor)

# 5th bottom

img28_ravel_all=[]
for i in range(Num):
  img28_ravel=np.copy(img28_all[i].ravel())
  img28_ravel_all.append(img28_ravel)
  # img28_ravel_all.append(img28_all[i].ravel())

# 6th bottom

img28_bottom=pd.DataFrame(img28_ravel_all)
img28_bottom.insert(0,"Type",Types_bottom)
img28_bottom.insert(0, "Width", SizeWidth) 
#print(img28_bottom)

frames = [img28_top,img28_middle,img28_bottom]
img28_all=pd.concat(frames)
#print(img28_all)

# found drive
from google.colab import drive
drive.mount('drive')

ww[11].shape

img28_all.to_csv('img28_all00.csv',float_format="%.5f")
# save in drive
!cp img28_all00.csv drive/My\ Drive/Maria_Gabriela_Textura_dados_jan_2020/

InputAnn=img28_all[['Type','Width']]
Choice=img28_all[img28_all.Type.isin(['B','G'])]
print(Choice)

X_fake=Choice[['Type','Width']]
print(X_fake)