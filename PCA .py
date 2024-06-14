#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,r2_score
plt.style.use ("dark_background")

import os 


# In[15]:


rng = np.random.RandomState(1)
X = np.dot(rng.rand(2,2), rng.randn(2,200)).T
plt.scatter(X[:,0], X[:,1])
plt.axis('equal');


# In[16]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)


# In[17]:


print(pca.components_)


# In[18]:


print(pca.explained_variance_)


# In[20]:


def draw_vector(vo, v1, ax=None):
    ax= ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                   linewidth=2,
                   shrinkA=0, shrinkB=0)
    ax.annotate('',v1,vo, arrowprops=arrowprops)
    
#plt data

plt.scatter(X[:,0], X[:,1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector *3* np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_+ v)
    plt.axis('equal');


# In[21]:


pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print('original shape :', X.shape)
print('Transformed shape :', X_pca.shape)


# In[22]:


X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:,0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:,0], X_new[:, 1], color = 'b', alpha = 0.8)
plt.axis('equal');


# In[23]:


from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape


# In[24]:


pca = PCA(2) #Project from 64 to 2 dimensions
pca.fit(digits.data)
projected = pca.transform(digits.data)
print(digits.data.shape)
print(projected.shape)


# In[25]:


plt.scatter(projected[:,0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral',10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();


# In[27]:


#choosing the number of component
pca=PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[ ]:




