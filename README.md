# Heart Diseases Prediction
## 1. Overview
The purpose of this project is to predict whether the patients diagnosed of heart disease or not. This project was conducted based on the dataset from https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset and created using Spyder IDE. 
## 2. Methodology
### Data processing
The data is split into train test set with a ratio of 70:30
### Model
The classification problem was handled by using feedforward neural network.
Model: "sequential"
![image](https://user-images.githubusercontent.com/108311968/184264741-3079a194-c4db-43db-a01b-9196d028ccd2.png)

The created model with both training and validation accuracy above 90% and difference between training and validation loss should be <15% to avoid over/underfitting
Thus, the model is trained with batch size of 20 in 40 epochs. 
### 3. Outcome
This model obtain accuracy of 99.86% with validation accuracy of 100%

![image](https://user-images.githubusercontent.com/108311968/184266190-23c833f0-ff56-4551-b8c5-06fc335fe2c1.png)
![image](https://user-images.githubusercontent.com/108311968/184266211-bef16e8f-6a93-4272-8212-1534ee906d03.png)
