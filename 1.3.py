# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:11:22 2020

@author: Mohammad
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
# =============================================================================
# X=2*np.random.random((100,2))-1
# 
# Y=np.zeros((len(X),1))
# for i in range(len(X)):
#     if(X[i,0]*X[i,0]+X[i,1]*X[i,1]<=1):
#         Y[i]=1
#     else:
#         Y[i]=-1
# =============================================================================
def plot_model(model):
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=50, cmap='autumn')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1], 200)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = model.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.show()

    y1_model = model.predict(X_train)
    y2_model = model.predict(X_test)

    print('Accuracy on train data:',accuracy_score(Y_train, y1_model))
    print('Accuracy on test data:',accuracy_score(Y_test, y2_model))

X,Y=make_circles(n_samples=200,noise=0.05,factor=0.5)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
svclassifier = svm.SVC(kernel='rbf',C=1E7)
average=cross_val_score(svclassifier,X_train,Y_train,cv=5).mean()
svclassifier.fit(X_train, Y_train)
plot_model(svclassifier)
print('validaton accuracy rbf : ' , average)
print('accuracy test rbf = ' ,accuracy_score(Y_test , svclassifier.predict(X_test)))
print('accuracy train rbf  = ' ,accuracy_score(Y_train , svclassifier.predict(X_train)))
svclassifier = svm.SVC(kernel='poly',degree=2,C=1E7)
average=cross_val_score(svclassifier,X_train,Y_train,cv=5).mean()

svclassifier.fit(X_train, Y_train)
plot_model(svclassifier)
print('accuracy test degree 2  = ' ,accuracy_score(Y_test , svclassifier.predict(X_test)))
print('accuracy train degree 2 = ' ,accuracy_score(Y_train , svclassifier.predict(X_train)))
print('validaton accuracy degree 2 : ' , average)

svclassifier = svm.SVC(kernel='poly',degree=3,C=1E3)
average=cross_val_score(svclassifier,X_train,Y_train,cv=5).mean()
svclassifier.fit(X_train, Y_train)
plot_model(svclassifier)
print('validaton accuracy degree 3 : ' , average)
print('accuracy test degree 3 = ' ,accuracy_score(Y_test , svclassifier.predict(X_test)))
print('accuracy train degree 3= ' ,accuracy_score(Y_train , svclassifier.predict(X_train)))