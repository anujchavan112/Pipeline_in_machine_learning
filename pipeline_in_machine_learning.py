#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
data = pd.read_csv(r'C:\Users\Anuj\Downloads\Project happiness\Project happiness\2019.csv')
data


# In[27]:


data.dtypes


# In[28]:


x=data['Overall rank']
y=data['GDP per capita']
plt.plot(x,y)


# In[29]:


x=data['Overall rank']
y=data['Social support']
plt.plot(x,y)


# In[30]:


x=data['Overall rank']
y=data['Score']
plt.plot(x,y)


# In[31]:


x=data['Country or region']== 'India' 
t=data[x]
t


# In[138]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


train = np.asarray(data.drop(['Country or region','Overall rank'],axis=1))
test=np.asarray(data['Score'])
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.33, random_state=42)


# In[139]:


model_pipeline_rfgr = Pipeline([('scaler1', StandardScaler()), 
                                 ('random_forest', RandomForestRegressor(max_depth=10,random_state=2))
                                 ])


# In[140]:


model_pipeline_lgr = Pipeline([('scaler2', StandardScaler()), 
                                 ('linearrgr', LinearRegression())
                                 ])


# In[142]:


model_pipeline_drgr = Pipeline([('scaler3', StandardScaler()), 
                                 ('distreergr', DecisionTreeRegressor(random_state=2))
                                 ])


# In[143]:


pipelines=[model_pipeline_rfgr,model_pipeline_lgr,model_pipeline_drgr]


# In[144]:


bestaccu=0.0
bestrgr=0
bestpipeline=""


# In[145]:


pipe_dict={0:"Random Forest regression",1:"Linear Regression",2:"Decesion Tree Regressor "}


# In[146]:


for pipe in pipelines:
    pipe.fit(X_train,y_train)


# In[147]:


for i,model in enumerate(pipelines):
    print("{} Test acuuracy:{}".format(pipe_dict[i],model.score(X_test,y_test)))


# In[148]:


model_pipeline_rfgr.predict(X_test)


# In[149]:


model_pipeline_drgr.predict(X_test)


# In[150]:


model_pipeline_lgr.predict(X_test)


# In[152]:


x=model_pipeline_drgr.predict(X_test)
y=y_test
plt.title("plot for Random Forest regression accuracy")
plt.plot(x)
plt.plot(y)
plt.show()


# In[153]:


x=model_pipeline_rfgr.predict(X_test)
y=y_test
plt.title("[plot for Decesion Tree Regressor accuracy]")
plt.plot(x)
plt.plot(y)
plt.show()


# In[154]:


x=model_pipeline_lgr.predict(X_test)
y=y_test
plt.title("[plot for Linear Regression accuracy]")
plt.plot(x)
plt.plot(y)
plt.show()


# In[ ]:


|

