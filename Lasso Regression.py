"""
Lasso Regression
Author: Balamurali M
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')

#Generating matrix with explanatory and response variable
matr = np.random.randint(200, size=(100, 5))
print (matr.shape)

train_exp = matr[:80, :4]
train_res = matr[:80, 4:]
test_exp = matr[80:, :4]
test_act = matr[80:, 4:]

print('train_exp',train_exp.shape)
print('train_res',train_res.shape)
print('test_exp',test_exp.shape)
print('test_act',test_act.shape)

#Lasso
ls = linear_model.Lasso(alpha=0.1) #alpha can be fine tuned
ls.fit(train_exp, train_res)
predicted1 = ls.predict(test_exp)
print("Lasso Predicted Values")
print (predicted1)
print ('Mean Square Error Lasso')
mse_1 = mean_squared_error(test_act, predicted1)  
print (mse_1)

#Linear Regression results
LR = LinearRegression()
LR.fit(train_exp, train_res)
predicted2 = LR.predict(test_exp)
print("Linear Regression Predicted Values")
print (predicted2)
print ('Mean Square Error Linear Reg')
mse_2 = mean_squared_error(test_act, predicted2)  
print (mse_2)     