#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn import datasets
iris = datasets.load_iris()
import scipy.stats as sp
import seaborn as sns


# In[2]:


#Импорт БД
Base = pd.read_csv('C:/Users/Dima/Desktop/АНДАН - workable data (1).csv', delimiter=',', 
                 names=['Имя', 'Цена', 'район', 'площадь', 'Цена за квадрат',
                        'кол-во комнат', 'тип дома', 'этажность', 'этаж', 'срок сдачи'])
print(Base)


# In[3]:


#Быстро взглянем на все значения в данных:
check = Base.describe()
print(check)


# In[4]:


'''Гипотиза 1:
    чем больше этажность дома, тем больше этаж на 
    котором находится квартира влияет на стоимость кВ м
'''

# Определим список моделей, которые хотим рассмотреть:
models = ["этажность", "этаж", "Цена за квадрат"]


# In[5]:


# Создадим копию данных только с 8 ведущими производителями:
data = Base[models].describe()


# In[6]:


# Поиск необходимых данных для создания таблицы
qu   = data.quantile([0.25, 0.75], interpolation='nearest') # квартили
med  = data.median() # медиана
skew = data.skew() # асимметрия
kurt = data.kurt() # эксцесс
StageH = data['этажность']
StageF = data['этаж']
PbS    = data['Цена за квадрат']


# In[7]:


# Создаем дата фрейм
Tabl1 = {'Статистика': ['среднее', 'медиана', 'Станд. откл.', 
                       'Вверхний квартиль', 'нижний квартиль',
                      'кол-во наблюдений', 'кол-во пропусков',
                      'минимальное знач.', 'максимальное знач.', 'Эксцесс', 'Асимметрия'], 
         'Этажность' : [StageH[1], med[0], StageH[2], 17.000000, 
                        6.545369, StageH[0], 0, StageH[4], StageH[7], kurt[0], skew[0]
                       ],
         'этаж' : [StageF[1], med[1], StageF[2], 8.0, 4.0, StageF[0],
                        0, StageF[4], StageF[7], kurt[1], skew[1]
                  ],
         'цена за квадрат': [PbS[1], med[2], PbS[2], 90809.820248,
                             49758.000000, PbS[0], 0, PbS[4], PbS[7], kurt[2], skew[2]
         ]}

Tabl1 = pd.DataFrame(Tabl1)
print(Tabl1)


# In[8]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=Tabl1.values, colLabels=Tabl1.columns, loc='center')

fig.tight_layout()

plt.show()


# In[9]:


'''Гипотеза 2:
    Площадь квартиры наибольшим образом влияет на
    ее стоимость, однако если квартира расположена
    в центре города площадь квартиры может уменьшаться,
    а стоимость возрастать
'''
# Определим список моделей, которые хотим рассмотреть:
models = ["район", "площадь", "Цена за квадрат"]


# In[10]:


# Создадим копию данных только с 8 ведущими производителями:
data = Base[models].describe()
print(data)


# In[11]:


# Поиск необходимых данных для создания таблицы
qu   = data.quantile([0.25, 0.75], interpolation='nearest') # квартили
med  = data.median() # медиана
skew = data.skew() # асимметрия
kurt = data.kurt() # эксцесс
Sq = data['площадь']
PbS    = data['Цена за квадрат']
print(med)


# In[12]:


# Создаем дата фрейм
Tabl2 = {'Статистика': ['среднее', 'медиана', 'Станд. откл.', 
                       'Вверхний квартиль', 'нижний квартиль',
                      'кол-во наблюдений', 'кол-во пропусков',
                      'минимальное знач.', 'максимальное знач.', 'Эксцесс', 'Асимметрия'], 
         'площадь' : [Sq[1], med[0], Sq[2], 4669.672727, 53.700000, Sq[0],
                        0, Sq[4], Sq[7], kurt[0], skew[0]
                  ],
         'цена за квадрат': [PbS[1], med[1], PbS[2], 90809.820248,
                             49758.000000, PbS[0], 0, PbS[4], PbS[7], kurt[1], skew[1]
         ]}

Tabl2 = pd.DataFrame(Tabl2)
print(Tabl2)


# In[13]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=Tabl2.values, colLabels=Tabl2.columns, loc='center')

fig.tight_layout()

plt.show()


# In[14]:


'''Гипотеза 3:
    Стоимость квартиры зависит от количества комнат
'''


# In[15]:


# Определим список моделей, которые хотим рассмотреть:
models = ["кол-во комнат", "Цена за квадрат"]


# In[16]:


# Создадим копию данных только с 8 ведущими производителями:
data = Base[models].describe()
print(data)


# In[17]:


# Поиск необходимых данных для создания таблицы
qu   = data.quantile([0.25, 0.75], interpolation='nearest') # квартили
med  = data.median() # медиана
skew = data.skew() # асимметрия
kurt = data.kurt() # эксцесс
nRoom = data['кол-во комнат']
PbS    = data['Цена за квадрат']
print(qu)


# In[18]:


# Создаем дата фрейм
Tabl3 = {'Статистика': ['среднее', 'медиана', 'Станд. откл.', 
                       'Вверхний квартиль', 'нижний квартиль',
                      'кол-во наблюдений', 'кол-во пропусков',
                      'минимальное знач.', 'максимальное знач.', 'Эксцесс', 'Асимметрия'], 
         'кол-во комнат' : [nRoom[1], med[0], nRoom[2], 5.050165, 1.809917, nRoom[0],
                        0, nRoom[4], nRoom[7], kurt[0], skew[0]
                  ],
         'цена за квадрат': [PbS[1], med[1], PbS[2], 90809.820248,
                             49758.000000, PbS[0], 0, PbS[4], PbS[7], kurt[1], skew[1]
         ]}

Tabl3 = pd.DataFrame(Tabl3)
print(Tabl3)


# In[19]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=Tabl3.values, colLabels=Tabl3.columns, loc='center')

fig.tight_layout()

plt.show()


# In[20]:


# анализ качественных переменных
model = ['Цена', 'площадь', 'Цена за квадрат', 'кол-во комнат', 'этажность', 'этаж']


# In[21]:


data = Base[model]
analVal = data.describe()


# In[22]:


# Создаем дата фрейм
Tabl1 = {'Переменная': ['Pr', 'Sq', 'PbS', 'nRoom', 'StageH', 'StageF'],
         'cреднее' : [
             analVal[model[0]][1], analVal[model[1]][1], analVal[model[2]][1],
             analVal[model[3]][1], analVal[model[4]][1], analVal[model[5]][1]
                     ],
         'медиана' : [
             data[model[0]].median(), data[model[1]].median(),
             data[model[2]].median(), data[model[3]].median(),
             data[model[3]].median(), data[model[4]].median()
         ],
         'мин. знач.': [
             analVal[model[0]][3], analVal[model[1]][3], analVal[model[2]][3],
             analVal[model[3]][3], analVal[model[4]][3], analVal[model[5]][3]
         ],
         'макс. знач.' : [
             analVal[model[0]][7], analVal[model[1]][7], analVal[model[2]][7],
             analVal[model[3]][7], analVal[model[4]][7], analVal[model[5]][7]
         ],
         'стандартное отклонение' : [
             analVal[model[0]][2], analVal[model[1]][2], analVal[model[2]][2],
             analVal[model[3]][2], analVal[model[4]][2], analVal[model[5]][2]
         ],
         'вариация' : [
             data[model[0]].var(), data[model[1]].var(),
             data[model[2]].var(), data[model[3]].var(),
             data[model[3]].var(), data[model[4]].var()
         ]
        }

Tabl1 = pd.DataFrame(Tabl1)
print(Tabl1)


# In[23]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=Tabl1.values, colLabels=Tabl1.columns, loc='center')

fig.tight_layout()

plt.show()
plt.savefig("таблица характеристик переменных.svg")


# In[24]:


qu   = data.quantile([0.25, 0.75], interpolation='nearest')
skew = data.skew()
kurt


# In[51]:


Tabl1 = {'Переменная': ['Pr', 'Sq', 'PbS', 'nRoom', 'StageH', 'StageF'],
         'cреднее' : [
             round(analVal[model[0]][1], 3), round(analVal[model[1]][1], 3), 
             round(analVal[model[2]][1], 3), round(analVal[model[3]][1], 3), 
             round(analVal[model[4]][1], 3), round(analVal[model[5]][1], 3)
                     ],
         'медиана' : [
             round(data[model[0]].median(), 3), round(data[model[1]].median(), 3),
             round(data[model[2]].median(), 3), round(data[model[3]].median(), 3),
             round(data[model[3]].median(), 3), round(data[model[4]].median(), 3)
         ],
         'стандартное отклонение' : [
             round(analVal[model[0]][2], 3), round(analVal[model[1]][2], 3), 
             round(analVal[model[2]][1], 3), round(analVal[model[3]][2], 3), 
             round(analVal[model[4]][2], 3), round(analVal[model[5]][2], 3)
         ],
         'Вверхний квартиль' : [5150000, 68.4, 99500, 2, 17, 8],
         'нижний квартиль' : [3409000, 39.9, 79012, 1, 5, 2],
         'кол-во наблюдений' : [484, 484, 484, 484, 484, 484],
         'кол-во пропусков' : [0, 0, 0, 0, 0, 0],
         'минимальное знач.' : [
             analVal[model[0]][3], analVal[model[1]][3], analVal[model[2]][3],
             analVal[model[3]][3], analVal[model[4]][3], analVal[model[5]][3]],
         'максимальное знач.' : [
             analVal[model[0]][7], analVal[model[1]][7], analVal[model[2]][7],
             analVal[model[3]][7], analVal[model[4]][7], analVal[model[5]][7]
         ],
         'Эксцесс' : [kurt[0].round(3), kurt[0].round(3), kurt[0].round(3),
                     kurt[0].round(3), kurt[0].round(3), kurt[0].round(3)],
         'Асимметрия' : [skew[0].round(3), skew[1].round(3), skew[2].round(3),
                     skew[3].round(3), skew[4].round(3), skew[5].round(3)]
        }

Tabl1 = pd.DataFrame(Tabl1)
print(Tabl1)


# In[53]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

tab = ax.table(cellText=Tabl1.values, colLabels=Tabl1.columns, loc='center')
tab.auto_set_font_size(False)
tab.scale(1, 3)
tab.set_fontsize(9)

fig.tight_layout()

plt.show()
plt.savefig("таблица характеристик переменных.svg")


# In[ ]:




