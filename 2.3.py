# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 17:17:41 2020

@author: Mohammad
"""
from sklearn.datasets import load_iris 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,StratifiedKFold

X,Y=load_iris(return_X_y=True)

Kfold=StratifiedKFold(n_splits=4, shuffle=True)
AV_R=0
AV_K2=0
AV_K3=0
AVERAGE=[0,0,0]
for train,test in Kfold.split(X,Y):
    X_train=X[train]
    Y_train=Y[train]
    X_test=X[test]
    Y_test=Y[test]
    svclassifier = svm.SVC(kernel='linear',decision_function_shape='ovr')

    
    svclassifier = svm.SVC(kernel='rbf' ,C=1E7)
    average=cross_val_score(svclassifier,X_train,Y_train,cv=4).mean()
    svclassifier.fit(X_train, Y_train)
    AV_R=average
    AVERAGE[0]=AVERAGE[0]+average
    print('validaton accuracy degree rbf : ' , average)
    print('number of support vectors for rbf=' ,len(svclassifier.support_vectors_))
    
    
    svclassifier = svm.SVC(kernel='poly',degree=2,C=1E7)
    average=cross_val_score(svclassifier,X_train,Y_train,cv=4).mean()
    AV_K2=average
    AVERAGE[1]=AVERAGE[1]+average
    svclassifier.fit(X_train, Y_train)
    print('validaton accuracy degree 2: ' , average)
    print('number of support vectors for degree 2 =' ,len(svclassifier.support_vectors_))
    svclassifier = svm.SVC(kernel='poly',degree=3,C=1E7)
    
    average=cross_val_score(svclassifier,X_train,Y_train,cv=4).mean()
    AV_K3=average
    AVERAGE[2]=AVERAGE[2]+average
    svclassifier.fit(X_train, Y_train)
    print('validaton accuracy degree 3  : ' , average)
    print('number of support vectors  for degree 3=' ,len(svclassifier.support_vectors_))
    A=[AV_R,AV_K2,AV_K3]
    print('accuracy test = ' ,accuracy_score(Y_test , svclassifier.predict(X_test)))
    print('accuracy train = ' ,accuracy_score(Y_train , svclassifier.predict(X_train)))
    
    if(A.index(max(A))==0):
        print('rbf have the most average accurcy')
        print('*_______________________________________*')
    if(A.index(max(A))==1):
        print('degree 2 have the most average accurcy ')
        print('*_______________________________________*')
    if(A.index(max(A))==2):
        print('degree 3 have the most average accurcy ')
        print('*_______________________________________*')
        
if(AVERAGE.index(max(AVERAGE))==0):
    print('rbf have the most average accurcy in total')
    print('*_______________________________________*')
if(AVERAGE.index(max(AVERAGE))==1):
    print('degree 2 have the most average accurcy in total ')
    print('*_______________________________________*')
if(AVERAGE.index(max(AVERAGE))==2):
    print('degree 3 have the most average accurcy in total ')
    print('*_______________________________________*')     