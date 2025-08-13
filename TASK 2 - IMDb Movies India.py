#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[7]:


# Step 2: Load Dataset (use correct path)
df = pd.read_csv("IMDb Movies India.csv", encoding="ISO-8859-1")


# In[8]:


# Step 3: Drop rows with missing ratings
df = df.dropna(subset=['Rating'])


# In[9]:


# Step 4: Clean 'Year' and 'Duration' columns
df['Year'] = df['Year'].str.extract(r'(\d{4})').astype(float)
df['Duration'] = df['Duration'].str.extract(r'(\d+)').astype(float)


# In[11]:


# Step 6: Use only top 10 frequent Directors and Genres
top_directors = df['Director'].value_counts().nlargest(10).index
df['Director'] = df['Director'].apply(lambda x: x if x in top_directors else 'Other')


# In[13]:


# Step 6: Fill missing values properly
df['Year'] = df['Year'].fillna(df['Year'].median())
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['Votes'] = df['Votes'].fillna(0)

# ✅ Create 'Main Genre' safely
df['Main Genre'] = df['Genre'].apply(lambda x: str(x).split(',')[0] if pd.notnull(x) else 'Unknown')

# ✅ Simplify Director and Actor columns
top_directors = df['Director'].value_counts().nlargest(10).index
df['Director'] = df['Director'].apply(lambda x: x if x in top_directors else 'Other')

top_actors = pd.concat([df['Actor 1'], df['Actor 2']]).value_counts().nlargest(10).index
df['Actor 1'] = df['Actor 1'].apply(lambda x: x if x in top_actors else 'Other')
df['Actor 2'] = df['Actor 2'].apply(lambda x: x if x in top_actors else 'Other')

# Step 7: Create model-ready DataFrame
df_model = df[['Year', 'Duration', 'Votes', 'Rating', 'Director', 'Main Genre', 'Actor 1', 'Actor 2']]
df_model = pd.get_dummies(df_model, columns=['Director', 'Main Genre', 'Actor 1', 'Actor 2'])


# In[14]:


# Step 9: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[15]:


# Step 10: Evaluate model
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# In[16]:


# Optional: Plot actual vs predicted
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Movie Ratings")
plt.grid()
plt.show()


# In[ ]:




