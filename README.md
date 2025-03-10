# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KIRAN G
RegisterNumber: 212223040095
*/

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```
```
dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
```
dataset.info()
```
```
X = dataset.iloc[:,:-1].values
print(x)
Y = dataset.iloc[:,1].values
print(y)
```
```
X.shape
```
```
Y.shape
```
```
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
X_train.shape
```
```
Y_train.shape
```
```
X_test.shape
```
```
Y_test.shape
```
```
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
```
```
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
```
plt.scatter(X_train, Y_train, color="red", label="Actual Scores")
plt.plot(X_train, reg.predict(X_train), color="blue", label="Fitted Line")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.show()
```
```
plt.scatter(X_test, Y_test, color='green', label="Actual Scores")
plt.plot(X_train, reg.predict(X_train), color='red', label="Fitted Line")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.show()
```

## Output:


![{795F9941-4D83-4795-91A2-8BC03297CE0A}](https://github.com/user-attachments/assets/784abd9e-590d-44d5-9185-04ec42e07bda)


![{BDD41522-905F-48AF-9ED6-0F2553413945}](https://github.com/user-attachments/assets/95b4b813-e817-4abc-9a5f-9b0e38805918)


![{9BEAFF2E-0578-462D-B67B-1EC035378ECE}](https://github.com/user-attachments/assets/a4c79434-5ec5-4534-a525-be90a36bc90a)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
