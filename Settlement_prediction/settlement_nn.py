# -*- coding: utf-8 -*-
""""
Original file is located at
    https://colab.research.google.com/drive/1TC-D7WGN1DDvzNasHM3-iVs51GTamBJS
Created by Peter D. Ogunjinmi; KNU, 2020
"""

# Import necessary libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import keras
from tensorflow import keras
from pylab import rcParams
from matplotlib import rc
%matplotlib inline
rcParams['figure.figsize'] = 10, 5

#Load dataset
data = pd.read_csv('Numerical_settlement_Pohang_Wkshop.csv', header=0)
data.head()

data.shape

# Show summary statistics of dataset
data.describe()

# Knowing The Data
# Correlation heatmap
corr = data.corr()
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.0) | (corr <= -0.0)],  vmax=1.0, vmin=-1.0, linewidths=0.1,
            xticklabels=True, yticklabels=True, annot = True, annot_kws={"size": 8}, cmap = 'coolwarm', square=True)
plt.title("Correlation Between Variables")
plt.savefig('2.png')

# pair Plot
sns.pairplot(data,palette="husl",diag_kind="kde")
plt.savefig('2.png')

# %matplotlib inline
data.hist(figsize=(15,10))
plt.show()

# input variables and target 
x_org = data.drop(['Settlement (mm)'], axis=1)
y_org = data['Settlement (mm)'].values


# Using Test/Train Split
from sklearn.model_selection import train_test_split
from numpy.random import seed

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(x_org,y_org, test_size=0.30, random_state=seed)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(-1,1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building ANN As a Regressor
from tensorflow import set_random_seed
set_random_seed(seed)
from keras.models import Sequential
from keras.layers import Dense
from keras import backend

#Defining Root Mean Square Error As our Metric Function 
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

from numpy.random import seed
seed(5)

# create model
model = Sequential()

#Building  first layer 
model.add(Dense(8,input_dim=3, kernel_initializer='glorot_normal', activation = 'sigmoid', name="dense_input" ))

# Output Layer
model.add(Dense(1, name="dense_output", activation='linear'))

#Optimize , Compile And Train The Model 
from keras import optimizers
opt = optimizers.Adam(lr=0.9)
model.compile(optimizer=opt,loss='mean_squared_error',metrics=[rmse])

history = model.fit(X_train,y_train,epochs = 60 ,batch_size=32,validation_split=0.1)
print(model.summary())

# Extract weights and biases from the input and hidden layers
weight_H1= pd.DataFrame(model.layers[0].get_weights())
weight_H1
model.layers[0].get_weights()
model.layers[1].get_weights()

# save model and architecture to single file
model.save("model_PI.h5")
print("Saved model to disk")

# Predicting and Finding R-Squared Score
y_predict = model.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_predict))
print(mean_absolute_error(y_test,y_predict))

# Plotting Loss And Root Mean Square Error For both Training And Test Sets
plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.title('Root Mean Squared Error')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('4.png')
plt.show()

#Plot scatter  of predictions
plt.suptitle('Scatter of predicitons on test data', fontsize=20)
plt.xlabel('True value', fontsize=18)
plt.ylabel('Predicted value', fontsize=16)
plt.scatter(y_test, y_predict)
plt.show()

# Write to csv format
data['Settlement'] = pd.Series(y_predict.reshape(1, -1)[0])
predicted_data = pd.concat([data['Settlement']])
predicted_data.to_csv('predicted_Norm_data.csv', index=False)


