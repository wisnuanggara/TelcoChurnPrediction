#ANN
#importing package

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Data Pre-proccessing
#Import the dataset
dataset = pd.read_csv('telco_Churn.csv')
dataset = dataset.drop(['customerID'], axis=1)
dataset['TotalCharges'] = dataset["TotalCharges"].replace(" ",np.nan)
dataset = dataset[dataset["TotalCharges"].notnull()]
dataset = dataset.reset_index()[dataset.columns]
dataset["TotalCharges"] = dataset["TotalCharges"].astype(float)

X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,19].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:,0] = labelencoder_X1.fit_transform(X[:,0])
labelencoder_X2 = LabelEncoder()
X[:,2] = labelencoder_X2.fit_transform(X[:,2])
labelencoder_X3 = LabelEncoder()
X[:,3] = labelencoder_X3.fit_transform(X[:,3])
labelencoder_X4 = LabelEncoder()
X[:,5] = labelencoder_X4.fit_transform(X[:,5])
labelencoder_X5 = LabelEncoder()
X[:,6] = labelencoder_X5.fit_transform(X[:,6])
labelencoder_X6 = LabelEncoder()
X[:,7] = labelencoder_X6.fit_transform(X[:,7])
labelencoder_X7 = LabelEncoder()
X[:,8] = labelencoder_X7.fit_transform(X[:,8])
labelencoder_X8 = LabelEncoder()
X[:,9] = labelencoder_X8.fit_transform(X[:,9])
labelencoder_X9 = LabelEncoder()
X[:,10] = labelencoder_X9.fit_transform(X[:,10])
labelencoder_X10 = LabelEncoder()
X[:,11] = labelencoder_X10.fit_transform(X[:,11])
labelencoder_X11 = LabelEncoder()
X[:,12] = labelencoder_X11.fit_transform(X[:,12])
labelencoder_X12 = LabelEncoder()
X[:,13] = labelencoder_X12.fit_transform(X[:,13])
labelencoder_X13 = LabelEncoder()
X[:,16] = labelencoder_X13.fit_transform(X[:,16])
labelencoder_X14 = LabelEncoder()
X[:,14] = labelencoder_X14.fit_transform(X[:,14])
labelencoder_X15 = LabelEncoder()
X[:,15] = labelencoder_X15.fit_transform(X[:,15])

X=pd.DataFrame(X)
dummy= pd.get_dummies(X[[6,7,8,9,10,11,12,13,16]], drop_first=True)
X=pd.concat([X,dummy],axis=1)
X=X.drop([6,7,8,9,10,11,12,13,16],axis=1)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y=pd.DataFrame(y)

X.to_csv('X_clean.csv',index=False)
y.to_csv('y_clean.csv',index=False)

