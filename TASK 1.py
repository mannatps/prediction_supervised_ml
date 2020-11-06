#!/usr/bin/env python
# coding: utf-8

# NAME : Mannatpreet Singh

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb


# In[2]:


df = pd.read_csv('data.csv')


# In[3]:


df.isna().sum()


# In[4]:


df.head(10)


# In[5]:


df.plot(x="Hours", y='Scores',style='o')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[6]:


x=df.iloc[:,:-1].values
y=df.iloc[:,1].values


# In[7]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[8]:


from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)

print("Training complete")


# In[9]:


line=regressor.coef_*x+regressor.intercept_
plt.scatter(x, y)
plt.plot(x,line)
plt.show()


# In[11]:


print(x_test)
y_pred=regressor.predict(x_test)


# In[12]:


df=pd.DataFrame({'Actual': y_test, 'Predict': y_pred})
df


# In[40]:


hours = 9.25
test=np.array([hours])
test=test.reshape(-1,1)
pred = regressor.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(pred[0]))


# In[42]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:', 
      metrics.mean_squared_error(y_test, y_pred)) 
print('RMS Error:', 
      np.sqrt(metrics.mean_squared_error(y_test,y_pred))) 


# In[ ]:




