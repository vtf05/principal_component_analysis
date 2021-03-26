
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv("Social_Network_Ads.csv")


X=dataset.iloc[:,2:4].values
Y=dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y , test_size=0.2,random_state=0)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)




from sklearn.decomposition import KernelPCA
kpca=KernelPCA(n_components=2 , kernel="rbf")
x_train= kpca.fit_transform(x_train)
x_test= kpca.transform(x_test)

from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)


y_pred = classifier.predict(x_test)


from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,y_pred)

 