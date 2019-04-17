#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[2]:


from collections import Counter


# In[3]:


import numpy as np


# In[4]:


messasge=[line.rstrip() for line in open('SMSSpamCollection')]
print(messasge)


# In[7]:


import pandas as pd


# In[8]:


len(messasge)


# In[9]:


#print(s,q=messasge[0].split('/t'))
df=pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])
df.head(5)


# In[10]:


df.describe()
    


# In[11]:


df.groupby('label').describe()


# In[12]:


df['length']=df['message'].apply(len)


# In[13]:


df.columns


# In[14]:


df.head()


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


df['length'].plot.hist(bins=50)


# In[17]:


df.hist(column='length',by='label',figsize=(12,4),bins=200)


# In[18]:


df[df['label']=='ham']['length'].describe()


# In[19]:


df[df['label']=='spam']['length'].describe()


# In[20]:


#df['no_stop']=[each for each in df['message'] if each.lower() not in stopwords.words('english')]
df.shape


# In[21]:


import string


# In[22]:


sp_w=string.punctuation
sp_w


# In[23]:


list_w=[]
list_ws=[]


# In[24]:


for i in df['message']:
    list_w=([each for each in i if each not in sp_w])
    
    list_w=''.join(list_w)
    print(list_w,'-----------')
    list_ws.append(list_w)
    


# Label Encoding for labels in dataset

# In[25]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['label']=le.fit_transform(df['label'])
df.head()


# Bag Of Word Creation

# In[26]:


#df['no_stop'].apply(len)
allwords=[]
for each in df['message']:
    
    eachs =each.split(' ')

    #allwords.append(eachs)
    allwords +=eachs


# In[28]:


dictionary=Counter(allwords)
list_of_word=dictionary.keys()
keyss=[]
for i in list_of_word:
    if i.isalpha()==False or len(i)<4:
        keyss.append(i)
    #if len(i)==1:
        #keyss.append(i)
#print(keyss)  
for i in keyss:
    dictionary.pop(i)
dictionary=dictionary.most_common(3000)    


# Feature Extraction

# In[29]:


features_matrix = np.zeros((5572,3000))
train_labels=np.zeros(5572)
docID=0
for i in range(df.shape[0]):
    #allwords=[]
    line=df['message'][i]
    words = line.split(' ')
    allwords=[]
    allword=Counter(words)
    allwords+=allword.keys()
    for word in allwords:
        wordID = 0
        for j,d in enumerate(dictionary):
            if d[0] == word:
                wordID = j
                features_matrix[docID][wordID] = words.count(word)
    train_labels[docID] = df['label'][i]
    docID = docID + 1      


# In[31]:


np.save('features_matrix.npy',features_matrix)
np.save('labels.npy',train_labels)


# Imbalanced Class Handling

# In[35]:


from imblearn.over_sampling import SMOTE


# In[36]:


train_matrix=np.load('features_matrix.npy')
labels=np.load('labels.npy')


# In[37]:


method=SMOTE(kind='regular')
x_s,y_s=method.fit_sample(train_matrix,labels)


# Train different model and compare the results

# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


train_x,test_x,train_y,test_y=train_test_split(x_s,y_s,test_size=0.30,random_state=0)


# In[42]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix


# In[43]:


N_B= MultinomialNB()
N_B.fit(train_x,train_y)
pred_y=N_B.predict(test_x)
print(confusion_matrix(test_y,pred_y))


# In[44]:


sv=LinearSVC()
sv.fit(train_x,train_y)
pred=sv.predict(test_x)
print(confusion_matrix(test_y,pred))


# In[45]:


from sklearn.linear_model import LogisticRegression


# In[46]:


le=LogisticRegression()
le.fit(train_x,train_y)
pp=le.predict(test_x)
print(confusion_matrix(test_y,pp))


# In[ ]:




