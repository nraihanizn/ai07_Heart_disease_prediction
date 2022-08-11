# -*- coding: utf-8 -*-
"""
Created on Mon Jul  25 10:54:46 2022

@author: MAKMAL2-PC23
"""

#1. Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn.datasets as skdatasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

#%%
#2. Data preparation
#load from sklearn
heart_disease = pd.read_csv(r"C:\Users\MAKMAL2-PC23\Documents\TensorFlow Deep Learning\project_ai07\heart.csv")
heart_disease.head()

#%%
#3. Split the data into features and label
x = heart_disease.drop('target',axis=1)
y = heart_disease['target']

#%%
#4. Perform train test split
SEED = 888
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=SEED)

#%%
#5. Perform data normalization
standardizer = StandardScaler()
standardizer.fit(x_train)
x_train= standardizer.transform(x_train)
x_test= standardizer.transform(x_test)

#%%
#6. Preparing the model
nClass = len(np.unique(np.array(y_test)))

model = keras.Sequential()

#input layer
model.add(layers.InputLayer(input_shape=(x_train.shape[1],)))
#Nthe hidden layer
model.add(layers.Dense(60, activation='relu'))
model.add(layers.Dense(30, activation='relu'))
#output layer
model.add(layers.Dense(nClass, activation='softmax'))

#%%
#Show the structure of the model
model.summary()

#%%
#7. Compile model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#%%
#8. Train model
BATCH_SIZE = 20
EPOCHS = 40
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=BATCH_SIZE,epochs=EPOCHS)

#%%
#9. Visualization
import matplotlib.pyplot as plt

training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_x_axis = history.epoch

plt.plot(epochs_x_axis,training_loss,label='Training Loss')
plt.plot(epochs_x_axis,val_loss,label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.legend()
plt.figure()

plt.plot(epochs_x_axis,training_acc,label='Training Accuracy')
plt.plot(epochs_x_axis,val_acc,label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.figure()

plt.show()

