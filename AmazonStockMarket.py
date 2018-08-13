
# coding: utf-8

# In[1]:


import pandas as pd
import quandl
import math, datetime
import time
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import style
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[2]:


style.use('ggplot')
df = quandl.get('SSE/AMZ',authtoken='_wr9eoMabhX7_bQTi2J9')
print(df.head())   


# In[3]:


print(df.tail())
df.plot(kind='box',subplots=True,layout=(1,5),sharex=False,sharey=False)
import matplotlib.pyplot as plt
plt.show()


# In[4]:


df.hist()
plt.show()


# In[5]:


scatter_matrix(df)
plt.show()
df['OC_Change']=(df['Last']-df['Previous Day Price'])/df['Previous Day Price']*100
df['HL_Change']=(df['High']-df['Low'])/df['Low']*100
df=df[['Last','HL_Change','OC_Change','Volume']]


# In[6]:


forecast_col='Last'
df.fillna(-9999, inplace=True)
forecast_out=int(math.ceil(0.01*len(df)))
df['label']=df[forecast_col].shift(-forecast_out)


# In[7]:


X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]


# In[8]:


df.dropna(inplace=True)
y=np.array(df['label'])


# In[9]:


X_train, X_test, y_train, y_test=cross_validation.train_test_split(X, y, test_size=0.2)

clf=LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy=clf.score(X_test,y_test)

print(accuracy)


# In[10]:


forecast_set=clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast']=np.nan


# In[11]:


last_date=df.iloc[-1].name
last_unix=last_date.timestamp()
one_day=86400
next_unix=last_unix+one_day


# In[12]:


for i in forecast_set:
	next_date=datetime.datetime.fromtimestamp(next_unix)
	next_unix+=one_day
	df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]


# In[13]:


df['Last'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

