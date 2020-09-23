# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 17:40:50 2020

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

for train,test in Kfold.split(X,Y):
    A=[]
    X_train=X[train]
    Y_train=Y[train]
    X_test=X[test]
    Y_test=Y[test]
    for i in range (400):
        svclassifier = svm.SVC(kernel='poly',degree=2,C=i+1)
        average=cross_val_score(svclassifier,X_train,Y_train,cv=4).mean()
        A.append(average)
        
    I=A.index(max(A))
    print(I,'\n')
    print('accuracy validation : ' , A[I])
    svclassifier = svm.SVC(kernel='poly',degree=2,C=I+1)
    svclassifier.fit(X_train, Y_train) 
    print('number of support vectors ' ,len(svclassifier.support_vectors_))
    print('accuracy test = ' ,accuracy_score(Y_test , svclassifier.predict(X_test)))
    print('accuracy train = ' ,accuracy_score(Y_train , svclassifier.predict(X_train)))
