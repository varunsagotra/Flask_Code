#!/usr/bin/env python
# coding: utf-8

# ## Currency_Notes_Authentication

# ### Data Set Info :: 
# 
# > Data were extracted from images that were taken from genuine and forged banknote-like specimens. 
# - For digitization, an industrial camera usually used for print inspection was used. 
# - The final images have 400x 400 pixels. 
# - Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. 
# - Wavelet Transform tool were used to extract features from images.

# In[1]:


import pandas as pd


# In[2]:


df1 = pd.read_csv("/Users/varunsagotra/Desktop/My_Projects/currency_notes_authentication.csv")
df1
# Feature : Class info :: 
#Class 0 means : Not Authentic
#Class 1 means : Authentic


# In[3]:


df1.info()
#Observation : No missing data in dataset


# `Info : As i am building this app to show docker steps, so not performing EDA on this dataset`

# In[4]:


X = df1.iloc[:,:-1] # Independent features
X.head(1)


# In[5]:


y = df1.iloc[:,-1] # Target feature : Class
y.head(1)


# In[6]:


print(X.shape)
print(y.shape)


# In[7]:


# Train Test Split
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain,ytest = train_test_split(X,y, test_size=0.3, random_state = 0)


# `Model Implementation : Random Forest Classifier <As its a classification probelm>`

# In[8]:


from sklearn.ensemble import RandomForestClassifier


# In[9]:


classifier = RandomForestClassifier()


# In[10]:


classifier.fit(Xtrain,ytrain)


# In[11]:


y_pred = classifier.predict(Xtest)


# In[12]:


# Check Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(ytest,y_pred)
# Observation : It might be an overfitting issue : Need to validate
# Since, i am concerned with model here so i am proceeding...


# `Export the Model to pickle file`

# In[13]:


import pickle
with open('currency_notes_authentication.pkl','wb') as f:
    pickle.dump(classifier,f)


# ### Validate the predicted result 

# In[14]:


import numpy as np


# In[15]:


classifier.predict([[1,2,1,2]])


# In[16]:


classifier.predict([[0,1,1,0]])


# In[ ]:




