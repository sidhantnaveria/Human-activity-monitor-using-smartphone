# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:02:51 2019

@author: sidhant
"""

import pandas as pd
import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils import plot_model
from keras.datasets import fashion_mnist
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Input,Flatten,Dropout,Activation
from sklearn.metrics import r2_score,confusion_matrix
from tensorflow.keras import regularizers
from keras.utils import np_utils
from keras.optimizers import SGD,Adam,Adamax
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
#
def metrices(data):
    data1=pd.DataFrame(columns=['Car','Still','Train','Walking','Bus','Precision','Recall','F1'])
    data1['Car']=data[0]
    data1['Still']=data[1]
    data1['Train']=data[2]
    data1['Walking']=data[3]
    data1['Bus']=data[4]
    precision=[]
    recall=[]
    sums=data.sum(axis=0)

#    print("preci",sums)
    for i in range(len(sums)):
        recall.append(data[i][i]/sums[i])
        

    
#    print("recall",recall)
    sum2=data.sum(axis=1)
    for i in range(len(sums)):
        precision.append(data[i][i]/sum2[i])
        

#    print(precision)
    data1['Precision']=precision
    data1['Recall']=recall
#    print(data1)
    data1['F1']=data1.F1.fillna(data1['Precision']*data1['Recall']/(data1['Precision']+data1['Recall']))
    
    return data1
    


def AE_method2(x_train,Ytrain):
    model = Sequential()
    input_size=x_train.shape[1]
    
    model.add(Dense(240,input_dim=input_size,activation='relu' ,kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(240,input_dim=input_size,activation='relu' ,kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(200,input_dim=input_size,activation='relu' ,kernel_regularizer=regularizers.l2(0.01)))
    
    model.add(Dropout(0.1))
    model.add(Dense(200,input_dim=input_size,activation='relu' ,kernel_regularizer=regularizers.l2(0.01)))
    
    model.add(Dropout(0.1))
#    model.add(Dense(32, activation='relu' ,kernel_regularizer=regularizers.l2(0.01)))
#    model.add(Dropout(0.3))
    model.add(Dense(200, activation='relu' ,kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.1))
#    sgd=SGD(lr=0.03,decay=1e-6,momentum=0.5,nesterov=True)
    adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)    
    
    model.add(Dense(5))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model
def model_randomForrest(X_train,y_train):
    rfclassifier = RandomForestClassifier(n_estimators=90, oob_score=True)
    rfclassifier.fit(X_train, y_train)
    return rfclassifier

data =pd.read_csv('C:\sidhant\CA3\Dataset\Data.csv')
#print(data.info())
data=data.drop(['android.sensor.accelerometer#min','android.sensor.accelerometer#max',
                'android.sensor.game_rotation_vector#min','android.sensor.game_rotation_vector#max',
                'android.sensor.gyroscope#min','android.sensor.gyroscope#max',
                'android.sensor.gyroscope_uncalibrated#min','android.sensor.gyroscope_uncalibrated#max',
                'android.sensor.linear_acceleration#min','android.sensor.linear_acceleration#max',
                'android.sensor.orientation#min','android.sensor.orientation#max',
                'android.sensor.rotation_vector#min','android.sensor.rotation_vector#max',
                'sound#min','sound#max','speed#min','speed#max'],axis=1)

data['target']=data['target'].astype('category')
data['target']=data['target'].cat.codes

data=data.drop('time',axis=1)
X=data.drop('target',axis=1)
y=data['target']
#dummy_y = np_utils.to_categorical(y)

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=0.30,random_state=4)
#Xtrain=Xtrain.values
#Xtest=Xtest.values
Xtrain=preprocessing.StandardScaler().fit_transform(Xtrain)
Xtest=preprocessing.StandardScaler().fit_transform(Xtest)


#
#model=AE_method2(Xtrain,Ytrain)
#model.fit(Xtrain, Ytrain, batch_size=40, epochs=70, validation_split=0.3)
#
#
#prediction=model.predict(Xtest)
#result=model.summary()
#result=tensorflow.keras.metrics.Accuracy(Ytest,prediction)
#result=r2_score(Ytest,prediction)
#print(result)

model=model_randomForrest(Xtrain,Ytrain)
prediction=model.predict(Xtest)
result=metrics.accuracy_score(Ytest,prediction)
#result=r2_score(Ytest,prediction)
conf_matrix=confusion_matrix(Ytest,prediction)
print(result)
data=pd.DataFrame(conf_matrix)
output= metrices(data)
print("confusion matrix with metrices for random forest:\n",output)



model= DecisionTreeClassifier(max_depth=19)
model.fit(Xtrain, Ytrain)
prediction=model.predict(Xtest)
result=metrics.accuracy_score(Ytest,prediction)
#result=r2_score(Ytest,prediction)
conf_matrix=confusion_matrix(Ytest,prediction)
print(result)
data=pd.DataFrame(conf_matrix)
output= metrices(data)
print("confusion matrix with metrices for decision tree:\n",output)

#model = GaussianNB()
#model.fit(Xtrain,Ytrain)
#prediction=model.predict(Xtest)
#result=metrics.accuracy_score(Ytest,prediction)
#print(result)
#conf_matrix=confusion_matrix(Ytest,prediction)
#data=pd.DataFrame(conf_matrix)
#output= metrices(data)
#print("confusion matrix with metrices for naive gaussian:\n",output)



svclassifier = SVC(kernel='rbf', gamma=0.09,C=22,tol=0.9)  
svclassifier.fit(Xtrain,Ytrain)
prediction=svclassifier.predict(Xtest)
print(svclassifier)
result1=metrics.accuracy_score(Ytest,prediction)

#result=r2_score(Ytest,prediction)
print(result1)
conf_matrix=confusion_matrix(Ytest,prediction)
data=pd.DataFrame(conf_matrix)
output= metrices(data)
print("confusion matrix with metrices for SVC:\n",output)

