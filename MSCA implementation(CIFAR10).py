#!/usr/bin/env python
# coding: utf-8

# In[62]:


#GROUP 10:
#Shaun Jacob Varghese: 20BAC10022
#Varun Ram S: 20BAC10038
#Manoshi Raha: 20BAC10020
#Jenish Murdia: 20BAC10004

import pandas as pd
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention
from keras.layers.merge import concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model


# In[63]:


from keras.datasets import cifar10
(trainx,trainy),(testx,testy) = cifar10.load_data()

print("Train of X and Y is:",(trainx.shape),(trainy.shape))


# In[64]:



from keras.datasets import cifar10
(trainx,trainy),(testx,testy) = cifar10.load_data()

print("Train of X and Y is:",(trainx.shape),(trainy.shape))


# In[65]:


trainx = trainx.astype('float32') 
testx = testx.astype('float32') 
trainx = trainx / 255.0 
testx = testx / 255.0

from keras.utils import np_utils 
trainy = np_utils.to_categorical(trainy) 
testy = np_utils.to_categorical(testy) 
num_classes = testy.shape[1]



# In[66]:


temp=Input(shape=(64,64,1))

c1=keras.layers.Conv2D(32, (7, 7), input_shape=(32,32,7), activation='relu', padding='same')(temp)
p1=keras.layers.MaxPool2D(pool_size=(2, 2))(c1)


x1=keras.layers.Conv2D(64,(1,1),input_shape=(64,64,1), activation='relu', padding='same')(p1)
x2=keras.layers.Conv2D(64,(3,3),input_shape=(64,64,3), activation='relu', padding='same')(p1)
x3=keras.layers.Conv2D(64,(5,5),input_shape=(64,64,5), activation='relu', padding='same')(p1)
x4=keras.layers.MaxPool2D(pool_size=(2,1))(p1)
x5=keras.layers.Conv2D(64,(1,1))(x4)


# In[67]:


from scipy.special import softmax


# In[68]:


#We will flatten after getting I1 to I4
y1=keras.layers.Flatten()(x1)
y2=keras.layers.Flatten()(x2)
y3=keras.layers.Flatten()(x3)
y5=keras.layers.Flatten()(x5)
#P(H,W,C) ppart of the research paper
par1_1=keras.layers.Dense(64,activation='relu')(y1)
par1_2=keras.layers.Dense(64,activation='relu')(y1)
par1_3=keras.layers.Dense(64,activation='relu')(y1)

par2_1=keras.layers.Dense(64,activation='relu')(y2)
par2_2=keras.layers.Dense(64,activation='relu')(y2)
par2_3=keras.layers.Dense(64,activation='relu')(y2)

par3_1=keras.layers.Dense(64,activation='relu')(y3)
par3_2=keras.layers.Dense(64,activation='relu')(y3)
par3_3=keras.layers.Dense(64,activation='relu')(y3)

par4_1=keras.layers.Dense(64,activation='relu')(y5)
par4_2=keras.layers.Dense(64,activation='relu')(y5)
par4_3=keras.layers.Dense(64,activation='relu')(y5)


# In[69]:


P1at = tf.matmul(par1_2,par1_3)
sm1_1 = tf.keras.activations.softmax(P1at, axis=-1)
sm1_2 = tf.matmul(par1_1,sm1_1)
#tf.keras.layers.Add(sm1_2,x1)

P2at = tf.matmul(par2_2,par2_3)
sm2_1 = tf.keras.activations.softmax(P2at, axis=-1)
sm2_2 = tf.matmul(par2_1,sm2_1)
#tf.keras.layers.Add(sm2_2,x2)

P3at = tf.matmul(par3_2,par3_3)
sm3_1 = tf.keras.activations.softmax(P3at, axis=-1)
sm3_2 = tf.matmul(par3_1,sm3_1)
#tf.keras.layers.Add(sm3_2,x3)

P4at =tf.matmul(par4_2,par4_3)
sm4_1 = tf.keras.activations.softmax(P4at, axis=-1)
sm4_2 = tf.matmul(par4_1,sm4_1)
#tf.keras.layers.Add(sm4_2,x4)

fin1=keras.layers.Flatten()(sm1_2)
#fin_1=tf.keras.layers.Add()([fin1,y1])
fin2=keras.layers.Flatten()(sm2_2)
fin3=keras.layers.Flatten()(sm3_2)
fin4=keras.layers.Flatten()(sm4_2)


# In[70]:


output = concatenate([fin1,fin2,fin3,fin4])
model = Model(inputs=temp, outputs=output)


# In[71]:


print(model.summary())


# In[61]:


model.save(r'C:\Users\shaun\Documents\Deep learning')


# In[ ]:




