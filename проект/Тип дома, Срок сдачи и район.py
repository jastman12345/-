#!/usr/bin/env python
# coding: utf-8

# In[9]:


#Импорт библиотек
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()
import seaborn as sns


# In[2]:


#Импорт БД
Base = pd.read_csv('C:/Users/Dima/Desktop/АНДАН - workable data (1).csv', delimiter=',', 
                 names=['Имя', 'Цена', 'район', 'площадь', 'Цена за квадрат',
                        'кол-во комнат', 'тип дома', 'этажность', 'этаж', 'срок сдачи'])


# In[ ]:


Tpe = Base['тип дома']
Ds  = Base['район']
Ln  = Base['срок сдачи']


# In[5]:


fig = plt.figure(figsize=(8, 6))
axs = fig.add_subplot(121)

n_bins2 = len(Tpe)

axs.set_title('Тип дома')
axs.tick_params(axis='x', rotation=70)

sns.histplot(Tpe, kde=True, color='orange')
#fig = sns_plot.get_figure()
plt.savefig("гистаграмма Типа дома.svg")


# In[6]:


fig = plt.figure(figsize=(8, 6))
axs = fig.add_subplot(121)

n_bins2 = len(Ds)

axs.set_title('Район')
axs.tick_params(axis='x', rotation=70)

sns.histplot(Ds, kde=True, color='orange')
#fig = sns_plot.get_figure()
plt.savefig("гистаграмма Районов.svg")


# In[8]:


fig = plt.figure(figsize=(8, 6))
axs = fig.add_subplot(121)

n_bins2 = len(Ln)

axs.set_title('Срок сдачи')
axs.tick_params(axis='x', rotation=90)

sns.histplot(Ln, kde=True, color='orange')
#fig = sns_plot.get_figure()
plt.savefig("гистаграмма сроков сдачи.svg")

