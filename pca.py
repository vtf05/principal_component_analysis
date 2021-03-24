import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Wine.csv")
X=dataset.iloc[:,0:13].values
Y=dataset.iloc[:,13].values


from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


#Applying Principal component analysis
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_varience = pca.explained_variance_ratio_


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(Y_test,y_pred)