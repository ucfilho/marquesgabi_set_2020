import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
import zipfile
from random import randint
from PIL import Image
import re
from sklearn.model_selection import train_test_split
#import scikit-image
import skimage
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

#!pip install mahotas

# Import the 'transform' module from 'skimage'
from skimage import transform

"""# Fourth step test ANN for data to be used for PSD (Particle size distribution) determination"""

# Mahotas used: marquesgabi_fev_2020/02_Mahotas_fracionado_fev_20_2020.ipynb

import mahotas.features.texture as mht
import mahotas.features
from skimage import filters
from skimage import exposure
import skimage.feature as sk
from google.colab import files
from numpy import linalg as LA
from scipy import stats
from scipy.signal import find_peaks
from scipy.signal import peak_prominences
from scipy.signal import peak_widths
from scipy import integrate
import re
import Go2BlackWhite
import Go2Mahotas

def Mahotas(Prop,df,Width_All):

    rows,cols=df.shape
    
    y=np.ones(rows,dtype=np.int8)
    
    Num=rows

    # select just the grains to show picture
    grain=[]
    Mahotas_Prop=[]
    Todas_Fotos=[] 

    img28_all=pd.DataFrame(df)
    Width_Grain_2=[]

    Size=28 # tamanho da foto
    Sub_Size=int(Size/5) # tamanho do fracionamento
    Row_Crop=1/2 # posicao do corte
    #Row_Crop=1/3 # posicao do corte
    Crop=int(Size*Row_Crop)


    for i in range(Num):
      if(y[i]==1):
        grain.append(i)
        Width_Grain_2.append(Width_All[i])

    cont=0 # 
    cols=5
    rows=int(len(grain)/cols)+1
    Grao_in_All28=[]

    for i in range(Num):
      if(y[i]==1):
        Grao_in_All28.append(i)
        cont=cont+1
        Foto=np.array(img28_all.iloc[i]).reshape(28,28)

        Prop_Escolhida=[]

        # p_foto=ww[k].reshape(Size,Size)
        p_foto=Foto
        GLCM=[]
        glcm_haralick=[]
        x_ref=[]
        Count=Sub_Size
        p=np.zeros((Sub_Size,Sub_Size))
        j_ref=0
        Cada_foto=[]
        Posicao_X=[]
        Posicao_Y=[]
        for k in range(Size):
          if((k+Sub_Size-1)<Size):
            #print("(k+Sub_Size)=",(k+Sub_Size),"k=",k)
            for i in range(Sub_Size):
              Posicao_X.append(Crop+i)
              for j in range(Sub_Size):
                p[i,j]=p_foto[Crop+i,j+k]
                Posicao_Y.append(j+k)

            WW=np.copy(p) 
            Cada_foto.append(WW.ravel())
            x_ref.append(Count-Sub_Size)
            Count=Count+1

            Mahotas =pd.DataFrame(mahotas.features.haralick(p.astype(int)), columns =Escolha)
            Prop_Escolhida.append(Mahotas[Prop].mean())

        Todas_Fotos.append(Prop_Escolhida)

    df_mahotas=pd.DataFrame(Todas_Fotos)



    Features_Total=[]
    cont=-1
    for i in range(Num):
      if(y[i]==1):
        cont=cont+1
        x_psd=df_mahotas.iloc[cont]
        peaks, rr = find_peaks(x_psd, height=0)

        N_peaks=len(peaks)
        prominences = peak_prominences(x_psd, peaks)

        #Area = simps(x, dx=1)
        Area = integrate.simps(x_psd, dx=1)
        if(len(peaks)==0):
          Width_peaks =0
          Width_peaks_max =0
          Width_peaks_min =0
          Media_proem=0    
        else:
          Width_peaks =np.mean(peak_widths(x_psd, peaks, rel_height=0.5))
          Width_peaks_max =np.max(peak_widths(x_psd, peaks, rel_height=0.5))
          Width_peaks_min =np.min(peak_widths(x_psd, peaks, rel_height=0.5))
          Media_proem=np.mean(prominences)
        Median = np.median(x_psd)
        Mode= stats.mode(x_psd)[0]
        Mean=np.mean(x_psd)
        Sd=np.std(x_psd)

        Features=[]
        Features.append(N_peaks)
        Features.append(Media_proem )
        Features.append(Area)
        Features.append(Width_peaks )
        Features.append(Width_peaks_max)
        Features.append(Width_peaks_min)
        Features.append(Median )
        Features.append(Mode[0])
        Features.append(Mean)
        Features.append(Sd)

        Features_Total.append(Features)

    Nomes_PSD=['N_peaks','Media_proem','Area','Width_peaks','Width_peaks_max',
                        'Width_peaks_min','Median','Mode','Mean','Sd'] 

    Features_Total=pd.DataFrame(Features_Total,columns=Nomes_PSD)



    rows=1;cols=2
    k=8

    Foto=np.array(img28_all.iloc[Grao_in_All28[k]]).reshape(28,28)


    Width_Grain=Width_All.iloc[grain]

    Width_Grain=np.array(Width_Grain) # passando de Serie (dataframe 1d) para np.array

    """# Fifth step create classes"""

    Width_bounds=[100,200]
    N_Class=4
    Class=[]
    a=Width_bounds[0]
    b=Width_bounds[1]
    delta_ab=(b-a)/N_Class
    for i in range(N_Class-1):
      valor=a+delta_ab*(i+1)
      Class.append(valor)

    Num=len(Width_Grain)
    count=[0,0,0,0]
    Hist_Width=[]
    for i in range(Num):
      if(Width_Grain[i]<Class[0]):
        count[0]=count[0]+1
        Hist_Width.append(0)
      elif(Width_Grain[i]<Class[1]):
        count[1]=count[1]+1
        Hist_Width.append(1)
      elif(Width_Grain[i]<Class[2]):
        count[2]=count[2]+1
        Hist_Width.append(2)
      else:
        count[3]=count[3]+1
        Hist_Width.append(3)


    N_count=np.copy(count)

    Nomes_class=['until_w1','w1-w2','w2-w3','bigger-w3']+Nomes_PSD


    rows=len(Width_Grain)
    cols=len(Nomes_PSD)
    Features_Class=np.zeros((4,cols)) # matrix containing average values stored. Each line is reprenting a different class

    for i in range(rows):
      k=Hist_Width[i]
      for j in range(cols):  
        Features_Class[k,j]=Features_Class[k,j]+ Features_Total.iloc[i,j]

    

    for i in range(4):
      if(N_count[i]==0):
        fator=0
      else:
        fator=1/N_count[i]
      Features_Class[i,:]=Features_Class[i,:]*fator
    

    df_mahotas_return=pd.DataFrame(Features_Class,columns=Nomes_PSD)
    
    return df_mahotas_return  #df_mahotas_return=Mahotas(Prop,df)

