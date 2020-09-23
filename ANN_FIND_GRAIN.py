import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
import zipfile
from random import randint
from PIL import Image
import re
from sklearn.model_selection import train_test_split
import skimage
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from skimage import transform

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# 03_ANN_NEW_DATA... only the grains in 882 are used for training 
#                    the ANN and segmented images are used to 
#                    train no-grain

"""# 03_ANN_NEW_DATA... only the grains in 882 are used for training the ANN and segmented images are used to train no-grain"""
def AnnGrain(df,df_class):
  y_valor=df['Type']

  quantidade= df.groupby('Type').size()

  df_G = df[df["Type"] == "G"] 
  Cut=['Unnamed: 0','Type','Width']
  FotosG= df_G.drop(Cut,axis=1)


  Size=28
  img_G=[]

  Num,cols=FotosG.shape
  for i in range(Num):
    data=np.array(FotosG.iloc[i]).reshape(Size,Size)
    img = Image.fromarray(data.astype('uint8'), mode='L')
    img=np.float32(img)
    img28=cv2.resize(img,(Size,Size), interpolation = cv2.INTER_AREA)
    img_G.append(img28)

  df_Z = df[df["Type"] == "Z"] 
  Cut=['Unnamed: 0','Type','Width']
  FotosZ= df_Z.drop(Cut,axis=1)

  # We'll choose which is grain and withdraw from 750 segmented photos

  Size=28
  img_Z=[]

  Num,cols=FotosZ.shape
  for i in range(Num):
    data=np.array(FotosZ.iloc[i]).reshape(Size,Size)
    img = Image.fromarray(data.astype('uint8'), mode='L')
    img=np.float32(img)
    img28=cv2.resize(img,(Size,Size), interpolation = cv2.INTER_AREA)
    img_Z.append(img28)

  GRAO=[0,146,149,166,217,222,223,257,268,286,455,482,538,612,644,647,651,677] # 0 ate 749
  GRAO=np.array(GRAO)
  Ind=FotosZ.index
  FotosNG=FotosZ.copy()
  for i in GRAO:
    FotosNG=FotosNG.drop(Ind[i])

  PERCENT=245.0/(len(FotosNG.index))
  FotosNG=FotosNG.sample(frac=PERCENT, replace=True)

  rows,col=FotosG.shape
  y_total=[] # grao-->zero, nao grao-->1
  for i in range(rows):
    y_total.append(0) #  # grao-->zero
  for i in range(rows,(2*rows)):
    y_total.append(1) #  # nao grao-->zero

  frames = [FotosG,FotosNG]
  result = pd.concat(frames)

  #Define data train and data test

  W_train, W_test, yw_train, yw_test = train_test_split(np.array(result), np.array(y_total), 
                                                      test_size=0.30, shuffle=True, 
                                                      random_state=42)

  train_images=W_train #imagens utilizadas para o treino
  train_labels=yw_train # resposta esperada para o treino
  test_images=W_test
  test_labels=yw_test

  model = keras.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  # GRAIN use crop photos other cases segmented
  model.fit(train_images, train_labels, epochs=200)

  #ANN das imagens
  x=np.array(W_test)
  logits = model(x, training=False)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  
  y_valor=np.copy(yw_test)
  data = {'y_Actual': y_valor,
          'y_Predicted': prediction
          }  # este dado esta no formato de dicionario

  df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])


  confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
  print(confusion_matrix)

  y_true = df['y_Actual']
  y_pred = df['y_Predicted']

  METRICS=sklearn.metrics.classification_report(y_true, y_pred)
  
  x_seg=np.array(df_class)
  logits = model(x_seg, training=False)
  y_pred= tf.argmax(logits, axis=1, output_type=tf.int32)
  y_pred= np.array(pd.DataFrame(y_pred))

  return y_pred,confusion_matrix,METRICS 
