import matplotlib.pyplot as plt
import numpy as np
import cv2
from random import randint
from PIL import Image
import re
from sklearn.model_selection import train_test_split
import skimage
import pandas as pd
import mahotas
import mahotas.features.texture as mht
import mahotas.features

# Commented out IPython magic to ensure Python compatibility.
#!git clone https://github.com/ucfilho/marquesgabi_fev_2020 #clonar do Github
#%cd marquesgabi_fev_2020

import Go2BlackWhite
import Go2Mahotas



# Segmentation: start here......

# start top
def Segmenta(img):
    #Start to use the big image
    img_ver=img.copy()
    Size=1200 # tamanho da foto
    Sub_Size=int(Size/5) # tamanho do fracionamento
    Row_Crop=1/10 # posicao do corte
    Crop=int(Size*Row_Crop)

    #FotoRotina=img.copy()
    #print('------START--------')
    #print(img)
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

    #3th top


    #4th top
    
    #TypesTop=[]

    #for i in range(Num):
      #Valor='Z'
      #TypesTop.append(Valor)
    


    # 5th top
    
    img28_ravel_all=[]
    for i in range(Num):
      img28_ravel=np.copy(img28_all[i].ravel())
      img28_ravel_all.append(img28_ravel)


    # 6th top

    img28_top=pd.DataFrame(img28_ravel_all)
    #img28_top.insert(0,"Type",TypesTop)
    img28_top.insert(0, "Width", SizeWidth)

    # Start middle

    Size=1200 # tamanho da foto
    Sub_Size=int(Size/5) # tamanho do fracionamento
    Row_Crop=1/2 # posicao do corte
    Crop=int(Size*Row_Crop)
   
    
    #First middle
    img=img_ver.copy()
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

    #3th middle

    #4th middle

    #Types_middle=[]
    #plt.figure(figsize=(15,15))
    for i in range(Num):
      #print('i=',i,'=====')
      Valor='Z'
      #Types_middle.append(Valor)

    # 5th middle

    img28_ravel_all=[]
    for i in range(Num):
      img28_ravel=np.copy(img28_all[i].ravel())
      img28_ravel_all.append(img28_ravel)
      # img28_ravel_all.append(img28_all[i].ravel())

    # 6th middle

    img28_middle=pd.DataFrame(img28_ravel_all)
    #img28_middle.insert(0,"Type",Types_middle)
    img28_middle.insert(0, "Width", SizeWidth)

    # start bottom

    #Start to use the big image
    Size=1200 # tamanho da foto
    Sub_Size=int(Size/5) # tamanho do fracionamento
    Row_Crop=9/10 # posicao do corte
    Crop=int(Size*Row_Crop)


    # First bottom
    img=img_ver.copy()

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


    # 3th bottom



    # 4th bottom

    #Types_bottom=[]
    #plt.figure(figsize=(15,15))
    for i in range(Num):

      Valor='Z'
      #Types_bottom.append(Valor)

    # 5th bottom

    img28_ravel_all=[]
    for i in range(Num):
      img28_ravel=np.copy(img28_all[i].ravel())
      img28_ravel_all.append(img28_ravel)

    # 6th bottom

    img28_bottom=pd.DataFrame(img28_ravel_all)
    #img28_bottom.insert(0,"Type",Types_bottom)
    img28_bottom.insert(0, "Width", SizeWidth) 

    frames = [img28_top,img28_middle,img28_bottom]
    img28_all=pd.concat(frames)
    

    return(img28_all)
