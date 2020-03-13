import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, interact_manual

#Import clean datasaet
X = pd.read_csv('X_clean.csv')
y = pd.read_csv('y_clean.csv')


#Splitting dataset into the training data and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialising ANN
classifier = Sequential()

#input layer and first hiden layer
classifier.add(Dense(output_dim=15, init ='uniform',activation='relu', input_dim=29))

#input second layer
classifier.add(Dense(output_dim=15, init ='uniform',activation='relu'))

#output layer
classifier.add(Dense(output_dim=1, init ='uniform',activation='sigmoid'))

#compile ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train, batch_size=10, nb_epoch=100)


#predict
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
