#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()


# In[2]:


#Импорт БД
Base = pd.read_csv('C:/Users/Dima/Desktop/АНДАН - workable data (1).csv', delimiter=',', 
                 names=['Имя', 'Цена', 'район', 'площадь', 'Цена за квадрат',
                        'кол-во комнат', 'тип дома', 'этажность', 'этаж', 'срок сдачи'])


# In[4]:


# Функция очистки от выбросов
def hampel(vals_orig):
    vals = vals_orig.copy()    
    difference = np.abs(vals.median()-vals)
    median_abs_deviation = difference.median()
    threshold = 3 * median_abs_deviation
    outlier_idx = difference > threshold
    vals[outlier_idx] = np.nan
    return(vals)


# In[5]:


Sq = hampel(Base['площадь'])
nRoom = hampel(Base['кол-во комнат'])


# In[8]:


min(Base['кол-во комнат'])


# In[6]:


#Диаграмма рассеивания для переменных цена за квартиру и площадь квартиры
fig, ax = plt.subplots()

ax.scatter(nRoom, Sq, c = 'deeppink')    

ax.set_title('Диаграмма рассеивания для переменных кол-во комнат и площадь', fontsize=32)
plt.xlabel('кол-во комнат', fontsize=32)
plt.ylabel('площадь', fontsize=32)

fig.set_figwidth(11.7 * 2)    
fig.set_figheight(8.27 * 2)   

plt.show()
plt.savefig("Диаграмма рассеивания для переменных кол-во комнат и площадь квартиры.svg")


# In[10]:


#Диаграмма рассеивания для переменных цена за квартиру и площадь квартиры
fig, ax = plt.subplots()

ax.scatter(Base['этажность'], Base['этаж'], c = 'deeppink')    

ax.set_title('Диаграмма рассеивания для переменных этажность и этаж', fontsize=32)
plt.xlabel('этажность', fontsize=32)
plt.ylabel('этаж', fontsize=32)

fig.set_figwidth(11.7 * 2)    
fig.set_figheight(8.27 * 2)   

plt.show()
plt.savefig("Диаграмма рассеивания для переменных этажность и этаж.svg")

