#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[33]:


df = pd.read_csv("Titanic-Dataset.csv")
df.head()


# In[34]:


df.info()
df.isnull().sum()


# In[35]:


if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)

for col in ['Name', 'Ticket']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].median())


if 'Fare' in df.columns:
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


# In[36]:


if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

if 'Embarked' in df.columns:
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

df.head()


# In[37]:


features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
if 'Embarked_Q' in df.columns:
    features.append('Embarked_Q')
if 'Embarked_S' in df.columns:
    features.append('Embarked_S')

X = df[features]
y = df['Survived']


# In[38]:


print("Missing values in X:\n", X.isnull().sum())


X = X.dropna()
y = y.loc[X.index] 


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[40]:


model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# In[41]:


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[42]:


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:




