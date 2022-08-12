# Heart Diseases Prediction
## 1. Overview
The purpose of this project is to predict whether the patients diagnosed of heart disease or not. This project was conducted based on the dataset from https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset and created using Spyder IDE. 
## 2. Methodology
### Data processing
The data is split into train test set with a ratio of 70:30
### Model
The classification problem was handled by using feedforward neural network.

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param   
=================================================================
 dense (Dense)               (None, 60)                840       
                                                                 
 dense_1 (Dense)             (None, 30)                1830      
                                                                 
 dense_2 (Dense)             (None, 2)                 62        
                                                                 
=================================================================
Total params: 2,732
Trainable params: 2,732
Non-trainable params: 0
_________________________________________________________________
