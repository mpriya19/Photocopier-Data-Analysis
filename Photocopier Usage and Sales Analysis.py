#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import requests

# setting seaborn
sns.set_palette('Spectral')
sns.set_context('notebook', font_scale=1)
sns.set_style('whitegrid')
sns.set(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"#FFF9ED"})

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


# read data
counter = pd.read_csv('E:/Projects/1. counter photocopy machine.csv')
transaction = pd.read_csv('E:/Projects/2. transaction item photocopy.csv')


# In[3]:


# check counter data type
counter.info()


# In[4]:


# transform date to datetime date type
counter['date'] = pd.to_datetime(counter['date']).dt.date


# In[5]:


# check transaction data type
transaction.info()


# In[6]:


# create function to transform price datatype to float
def transform_price(x):
    try:
        x = float(x.split(',')[0])
        msg = x
    except:
        None
    return msg

# apply the function
transaction['price'] = transaction['price'].apply(transform_price)

# transform date to datetime data type
transaction['date'] = pd.to_datetime(transaction['date']).dt.date


# In[7]:


# prepare the data
task1 = transaction.groupby(['date']).agg({'quantity':np.sum,'price':np.mean,'income':np.sum}).reset_index()
task1 = counter.merge(task1, how='left', on='date')
task1.dropna(axis=0, inplace=True)
task1 = task1[task1.index>=5]

# show 5 rows 
task1.head()


# In[12]:


# visualize the trend 
plt.figure(figsize=(15, 5))

plt.title('Trend of sheet sales compared to the number of counter machines', fontsize=20, pad=15)

sns.lineplot(task1, x='date', y='quantity', label='Quantity Sales', marker='o')
sns.lineplot(task1, x='date', y='counter', label='Counter Machine', marker='o')

plt.ylabel('Count of Sheets', fontsize=12, labelpad=15)
plt.xlabel('Date', fontsize=12, labelpad=15)

plt.tight_layout()
plt.show()

# visualize the correlation score
corr = task1.select_dtypes('number').corr()
bar = pd.DataFrame({
    'status':['Counter Machines','Quantity Sales'], 
    'count':[task1.counter.sum(), task1.quantity.sum()]
})

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))

# barrplot
ax[0].set_title('Grand total of sheet sales compared\nto the number of counter machines', pad=15, fontsize=15)
sns.barplot(bar, x='status', y='count', ax=ax[0])

# correlation
ax[1].set_title('Correlation Score\n', fontsize=15, pad=15)
sns.heatmap(corr, annot=True, linewidth=2, ax=ax[1])

plt.tight_layout()
plt.show()


# In[ ]:




