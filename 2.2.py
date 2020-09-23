"""
Created on Sun Jan  5 15:22:02 2020

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
    X_train=X[train]
    Y_train=Y[train]
    X_test=X[test]
    Y_test=Y[test]
    svclassifier = svm.SVC(kernel='linear',decision_function_shape='ovr')
    A=[]

    temp=0
    for i in range(1,400): 
        svclassifier.C=i+1
        average=cross_val_score(svclassifier,X_train,Y_train,cv=4).mean()
        A.append(average)

   
    I=A.index(max(A))
    xx=[i+1 for i in range (1,400)]
    Ma=A[I]
    plt.plot(xx,A)
    plt.show()
    plt.clf()
    svclassifier = svm.SVC(kernel='linear',C=I+1)
    svclassifier.fit(X_train,Y_train)
    print('accuracy test = ' ,accuracy_score(Y_test , svclassifier.predict(X_test)))
    print('accuracy  train = ' ,accuracy_score(Y_train , svclassifier.predict(X_train)))
    
    print ('accuracy= ' , Ma)
    print('number of support vectors =' ,len(svclassifier.support_vectors_))

    
