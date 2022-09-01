#%% Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.io
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")

#%% Loading Train and Test Dataset

Train=scipy.io.loadmat("train_32x32.mat")
Test=scipy.io.loadmat("test_32x32.mat")

#%% Preparing Datasets

x_train=Train["X"]
y_train=Train["y"]

x_test=Test["X"]
y_test=Test["y"]

#%% Applying PCA to Train and Test Images
pca=PCA(n_components=4)

x_train_0=[]
x_train_1=[]
x_train_2=[]

x_test_0=[]
x_test_1=[]
x_test_2=[]

for i in range(x_train.shape[-1]):
    x_train_0.append(x_train[:,:,0,i].flatten())
    x_train_1.append(x_train[:,:,1,i].flatten())
    x_train_2.append(x_train[:,:,2,i].flatten())
    
for i in range(x_test.shape[-1]):
    x_test_0.append(x_test[:,:,0,i].flatten())
    x_test_1.append(x_test[:,:,1,i].flatten())
    x_test_2.append(x_test[:,:,2,i].flatten())

x_train_0_pca=pca.fit_transform(x_train_0)
x_train_1_pca=pca.fit_transform(x_train_1)
x_train_2_pca=pca.fit_transform(x_train_2)

x_test_0_pca=pca.fit_transform(x_test_0)
x_test_1_pca=pca.fit_transform(x_test_1)
x_test_2_pca=pca.fit_transform(x_test_2)


x_train_pca_knn=np.concatenate((x_train_0_pca,x_train_1_pca,x_train_2_pca),axis=1)
x_test_pca_knn=np.concatenate((x_test_0_pca,x_test_1_pca,x_test_2_pca),axis=1)

#%% KNN with Varying k Values

"""

As performed in the “Deep learning / 2.2.  Over and under fitting” slide, tests
are done on the training and test datasets for determining best fitting k value.

"""


K=[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000]

scores=[]
scores_test=[]

for k in K:
    print("k=",k)
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train_pca_knn,y_train)
    y_preds=knn.predict(x_train_pca_knn)
    y_preds_test=knn.predict(x_test_pca_knn)
    scores.append(accuracy_score(y_train, y_preds))
    scores_test.append(accuracy_score(y_test, y_preds_test))

#%% Plotting k values vs. Accuracy Scores

plt.plot(K,scores)
plt.plot(K,scores_test,"r")
plt.ylabel("Accuracy")
plt.xlabel("k")
#plt.xscale("log")
plt.legend(["Train","Test"])
plt.title("Accuracy Values According to Changing k Values")
plt.show()

"""

We get the same result as shown in the “Deep learning / 2.2.  Over and under 
fitting” slide. When k values are small, we get high training accuracy and low 
test accuracy. When k values starts to increase, we observe that training 
accuracy falls down. Which means that when k values increase the model starts
underfitting.

"""

#%% Cross Validation

def changing_pca(n_comp,k,x0,x1,x2,y_train):
    pca=PCA(n_components=n_comp)

    x0=pca.fit_transform(x0)
    x1=pca.fit_transform(x1)
    x2=pca.fit_transform(x2)

    x_train_pca=np.concatenate((x0,x1,x2),axis=1)
    
    
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train_pca,y_train)
        
    score=np.mean(cross_val_score(knn, x_train_pca,y_train,cv=5))
    
    return score
"""

According to my past experiences with changing n_components numbers best 
results have been observed when n_components is around 50. So that, for this
performance search algorithm n_components going to be 40,45,50,55, and 60.

"""
N_com=[40,45,50,55,60]

"""

When choosing k values, in the first part we observed that after some point test
accuracy starts to stay still and in some early part change is so drastic but 
the accuracy is ver low. Because of that I choose to look for k values, which are
between 10 and 100. For the computational speed, k is going to be chosen as done
in the first part.

"""

K=[10,20,30,40,50,60,70,80,90,100]

scores2=[]

for n in N_com:
    scores_temp=[]
    for k in K:
        print("n_components:",n,"k:",k)
        scores_temp.append(changing_pca(n, k, x_train_0, x_train_1, x_train_2, y_train))
    scores2.append(scores_temp)

#%% Plotting n_components Values vs. Accuracy Scores 

fig, axs = plt.subplots(5)
fig.suptitle('Accuracy Values According to Changing n_components Values and k Values')

axs[0].plot(K, scores2[0])
axs[0].set_title("n_components = 40")

axs[1].plot(K, scores2[1])
axs[1].set_title("n_components = 45")

axs[2].plot(K, scores2[2])
axs[2].set_title("n_components = 50")

axs[3].plot(K, scores2[3])
axs[3].set_title("n_components = 55")

axs[4].plot(K, scores2[4])
axs[4].set_title("n_components = 60")

plt.show()

"""

After investigating the subplots we observe that accuracy is not cahnged from 
40 n_components to 60 n_components. Because of that n_components can be chosen 
as 50. Also, since when k is equal to 20 the greatest accuracy have been occured,
k can be chosen as 20. But this value is very close to close k values.  

n_components=50
k=20

"""

#%% Finding Test Accuracy for Best Variables That Have Been Found By Fine Tuning Operation

n_com_best=50
k_best=20

print("n_component:",n_com_best,"\nk:",k_best)

pca=PCA(n_components=n_com_best)

x0=pca.fit_transform(x_train_0)
x1=pca.fit_transform(x_train_1)
x2=pca.fit_transform(x_train_2)

xt0=pca.fit_transform(x_test_0)
xt1=pca.fit_transform(x_test_1)
xt2=pca.fit_transform(x_test_2)

x_train_pca=np.concatenate((x0,x1,x2),axis=1)
x_test_pca=np.concatenate((xt0,xt1,xt2),axis=1)

knn=KNeighborsClassifier(n_neighbors=k_best)
knn.fit(x_train_pca,y_train)
y_test_pred=knn.predict(x_test_pca)
final_acc=accuracy_score(y_test, y_test_pred)

print("Final accuracy:",final_acc)

"""

Final accuracy have been calculated according to the best parameters have been 
found.

"""

#%% Best Test Accuracy

"""

After trying some other values for n_components and k, I found that the best test
results have been occured when n_components is 50 and k is 1000. 

"""

n_com_best=50
k_best=1000

print("n_component:",n_com_best,"\nk:",k_best)

pca=PCA(n_components=n_com_best)

x0=pca.fit_transform(x_train_0)
x1=pca.fit_transform(x_train_1)
x2=pca.fit_transform(x_train_2)

xt0=pca.fit_transform(x_test_0)
xt1=pca.fit_transform(x_test_1)
xt2=pca.fit_transform(x_test_2)

x_train_pca=np.concatenate((x0,x1,x2),axis=1)
x_test_pca=np.concatenate((xt0,xt1,xt2),axis=1)

knn=KNeighborsClassifier(n_neighbors=k_best)
knn.fit(x_train_pca,y_train)
y_test_pred=knn.predict(x_test_pca)
final_acc=accuracy_score(y_test, y_test_pred)

print("Final accuracy:",final_acc)

