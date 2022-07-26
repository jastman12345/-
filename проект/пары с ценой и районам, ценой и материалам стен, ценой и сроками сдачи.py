#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()
import seaborn as sns


# In[29]:


#Импорт БД
Base = pd.read_csv('C:/Users/Dima/Desktop/АНДАН - workable data (1).csv', delimiter=',', 
                 names=['Имя', 'Цена', 'район', 'площадь', 'Цена за квадрат',
                        'кол-во комнат', 'тип дома', 'этажность', 'этаж', 'срок сдачи'])


# In[30]:


Base


# In[31]:


# Функция очистки от выбросов
def hampel(vals_orig):
    vals = vals_orig.copy()    
    difference = np.abs(vals.median()-vals)
    median_abs_deviation = difference.median()
    threshold = 3 * median_abs_deviation
    outlier_idx = difference > threshold
    vals[outlier_idx] = np.nan
    return(vals)


# In[32]:


# Необходимая переменная
Pr     = hampel(Base['Цена'])


# In[37]:


bs = Base.loc[Base['Цена'] == Pr]
bs = bs[bs['Цена'].notna()]


# In[38]:


#Категорированная диаграмма Бокса-Уискера
fig, ax = plt.subplots()

plt.ticklabel_format(axis='y', useOffset=False, style='plain')
sns.boxplot(data=bs, x='тип дома', y='Цена')

plt.title('Категорированная диаграмма Бокса-Уискера', fontsize=32)
plt.xlabel('тип дома', fontsize=32)
plt.ylabel('Цена', fontsize=32)
fig.set_size_inches(11.7, 8.27)

plt.show()
plt.savefig("Категорированная диаграмма Бокса-Уискера тип здания и цена.svg")


# In[39]:


#Категорированная диаграмма Бокса-Уискера
fig, ax = plt.subplots()

sns.boxplot(data=bs, x='район', y='Цена')

plt.ticklabel_format(axis='y', useOffset=False, style='plain')
plt.title('Категорированная диаграмма Бокса-Уискера', fontsize=32)
plt.xlabel('район', fontsize=32)
plt.ylabel('Цена', fontsize=32)
fig.set_size_inches(11.7, 8.27)

plt.show()
plt.savefig("Категорированная диаграмма Бокса-Уискера район и цена.svg")


# In[40]:


#Категорированная диаграмма Бокса-Уискера
fig, ax = plt.subplots()

sns.boxplot(data=bs, x='срок сдачи', y='Цена')

plt.ticklabel_format(axis='y', useOffset=False, style='plain')
plt.title('Категорированная диаграмма Бокса-Уискера', fontsize=32)
plt.xlabel('срок сдачи', fontsize=32)
plt.ylabel('Цена', fontsize=32)
plt.tick_params(axis='x', rotation=90)
fig.set_size_inches(11.7, 8.27)

plt.show()
plt.savefig("Категорированная диаграмма Бокса-Уискера срок сдачи и цена.svg")


# In[ ]:




