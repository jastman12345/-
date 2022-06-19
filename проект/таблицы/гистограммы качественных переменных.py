#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn import datasets
iris = datasets.load_iris()
import scipy.stats as sp
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# In[2]:


#Импорт БД
Base = pd.read_csv('C:/Users/Dima/Desktop/АНДАН - workable data (1).csv', delimiter=',', 
                 names=['Имя', 'Цена', 'район', 'площадь', 'Цена за квадрат',
                        'кол-во комнат', 'тип дома', 'этажность', 'этаж', 'срок сдачи'])


# In[3]:


# Функция очистки от выбросов
def hampel(vals_orig):
    vals = vals_orig.copy()    
    difference = np.abs(vals.median()-vals)
    median_abs_deviation = difference.median()
    threshold = 3 * median_abs_deviation
    outlier_idx = difference > threshold
    vals[outlier_idx] = np.nan
    return(vals)


# In[4]:


# Необходимые переменные
Pr     = hampel(Base['Цена'])
Sq     = hampel(Base['площадь'])
nRoom  = hampel(Base['кол-во комнат'])
PbS    = hampel(Base['Цена за квадрат'])
StageH = hampel(Base['этажность'])
StageF = hampel(Base['этаж'])


# In[5]:


#гистаграмма корреляций по цене
fig = plt.figure(figsize=(8, 6))
axs = fig.add_subplot(121)

n_bins1 = len(Pr)

axs.set_title('Цена')

sns.histplot(Pr, kde=True, color='orange')

plt.savefig("гистаграмма Цены.svg")


# In[6]:


#гистаграмма корреляций по площади
fig = plt.figure(figsize=(8, 6))
axs = fig.add_subplot(121)

n_bins1 = len(Base.loc[Base['площадь'].astype(np.float) < 1000]['площадь'])

axs.set_title('площадь')

sns.histplot(Base.loc[Base['площадь'].astype(np.float) < 1000]['площадь'], kde=True, color='orange')

plt.savefig("гистаграмма площади.svg")


# In[7]:


#гистаграмма корреляций по кол-ву комнат
fig = plt.figure(figsize=(8, 6))
axs = fig.add_subplot(121)

n_bins1 = len(Base.loc[Base['кол-во комнат'].astype(np.float) < 50]['кол-во комнат'])

axs.set_title('Цена за квадрат')

sns.histplot(Base.loc[Base['кол-во комнат'].astype(np.float) < 50]['кол-во комнат'], kde=True, color='orange')

plt.savefig("гистаграмма кол-во комнат.svg")


# In[8]:


#гистаграмма корреляций по этажности
fig = plt.figure(figsize=(8, 6))
axs = fig.add_subplot(121)

n_bins1 = len(StageH)

axs.set_title('этажность')

sns.histplot(StageH, kde=True, color='orange')

plt.savefig("гистаграмма этажности.svg")


# In[9]:


#гистаграмма корреляций по этажам
fig = plt.figure(figsize=(8, 6))
axs = fig.add_subplot(121)

n_bins1 = len(StageF)

axs.set_title('этажи')

sns.histplot(StageF, kde=True, color='orange')

plt.savefig("гистаграмма этажей на которых распологаются квартиры.svg")


# In[10]:


#гистаграмма корреляций по этажам
fig = plt.figure(figsize=(8, 6))
axs = fig.add_subplot(121)

n_bins1 = len(PbS)

axs.set_title('цена за квадрат')

sns.histplot(PbS, kde=True, color='orange')

plt.savefig("гистаграмма цены за квадрат.svg")


# In[ ]:




