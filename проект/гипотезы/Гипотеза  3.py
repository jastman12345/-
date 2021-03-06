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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from statsmodels.stats.diagnostic import het_white
import statsmodels.api as sm


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


#Гипотиза 3:
'''Стоимость квартиры зависит от количества комнат'''
PbSq    = Base['Цена за квадрат']
nRoomq  = Base['кол-во комнат']

PIR = sp.pearsonr(nRoomq, PbSq)
KT = sp.kendalltau(nRoomq, PbSq)
SP = sp.spearmanr(nRoomq, PbSq)


# In[6]:


Tab3 = {
    'пирсона' : PIR,
    'Спирмена' : SP,
    'Тау Кендалла' : KT
}

Tab = pd.DataFrame(Tab3)


# In[7]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

table = ax.table(cellText=Tab.values, colLabels=Tab.columns, loc='center')

fig.tight_layout()

plt.show()
plt.savefig("корреляции для гипотезы 3.svg")


# In[8]:


#'распределение квартир по\n количеству комнат
fig = plt.figure(figsize=(8, 6))
axs = fig.add_subplot(121)

X =  Base.loc[Base['кол-во комнат'].astype(np.float) < 50]['кол-во комнат']
n_bins1 = 5

axs.set_title('распределение квартир по\n количеству комнат', fontsize=16)

sns.histplot(X, kde=True, color='blue')
plt.savefig("распределение квартир по количеству комнат.svg")


# In[9]:


SuperBased = Base.copy()
SuperBased['кол-во комнат'] = hampel(SuperBased['кол-во комнат'])
SuperBased['Цена за квадрат'] = hampel(SuperBased['Цена за квадрат'])


# In[10]:


#Диаграмма рассеивания для переменных цена за квартиру и кол-во комнат
fig, ax = plt.subplots()

data = SuperBased['кол-во комнат']
y = SuperBased['Цена за квадрат']

ax.scatter(data, y, c = 'darkblue')    

ax.set_title('Диаграмма рассеивания для переменных цена за квартиру и кол-во комнат', fontsize=32)
plt.xlabel('кол-во комнат', fontsize=32)
plt.ylabel('Цена за квадратный метр', fontsize=32)

fig.set_figwidth(11.7 * 2)    
fig.set_figheight(8.27 * 2)   

plt.show()
plt.savefig("Диаграмма рассеивания для переменных цена за квартиру и кол-во комнат.svg")


# In[11]:


#Категорированная диаграмма Бокса-Уискера
fig, ax = plt.subplots()

sns.boxplot(data=SuperBased, x='кол-во комнат', y='Цена за квадрат')

plt.title('Категорированная диаграмма Бокса-Уискера', fontsize=32)
plt.xlabel('кол-во комнат', fontsize=32)
plt.ylabel('Цена за квадратный метр', fontsize=32)
plt.tick_params(axis='x', rotation=70)
ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
fig.set_size_inches(11.7 * 2, 8.27 * 2)

plt.show()
plt.savefig("Категорированная диаграмма Бокса-Уискера кол-во комнат и цена за квадрат.svg")


# In[12]:


kv = sp.stats.kruskal(nRoomq, PbSq)
print(kv)


# In[13]:


TabBox = {
    'Нулевая гипотеза' : ['Цена за квадратный метр зависит от кол-ва комнат'],
    'критерий' : ['критерий Краскела-Уоллиса'],
    'значимость' : [kv[1]],
    'значение статистики' : [kv[0]],
    'решение' : ['Нулевая гипотеза откланена']
}

Tab = pd.DataFrame(TabBox)


# In[14]:


fig, ax = plt.subplots(figsize=(15, 15))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

table = ax.table(cellText=Tab.values,
         colLabels=Tab.columns,
         loc='center',
         rowLoc='center'
        )

fig.tight_layout()

plt.show()
plt.savefig("таблица нулевой гипотезы для гипотезы 3.svg")


# In[15]:


nRoomq  = Base['кол-во комнат']
PbSq    = Base['Цена за квадрат']


# In[16]:


print(min(PbSq))


# In[17]:


slr = LinearRegression()

slr.fit(nRoomq.to_numpy().reshape(-1, 1), PbSq.to_numpy().reshape(-1, 1))

y_pred = slr.predict(nRoomq.to_numpy().reshape(-1, 1))

print(slr.coef_[0])
print(slr.intercept_)


# In[18]:


import statsmodels.formula.api as smf

results = smf.ols('PbSq ~ nRoomq', data=Base).fit()
print(results.summary())


# In[19]:


TabStat = {
    'переменная' : ['nRoom'],
    'Коэф.' : [results.params[1].round(3)],
    'станд. ошибка' : [results.bse[1].round(3)],
    't-статистика' : [results.tvalues[1].round(3)],
    'p-уровень' : [results.pvalues[1].round(3)],
    '95% дов интервал левый' : [results.conf_int()[0][1].round(3)],
    '95% дов интервал правый' : [results.conf_int()[1][1].round(3)]
}

TabStat = pd.DataFrame(TabStat)
print(TabStat)


# In[20]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=TabStat.values, colLabels=TabStat.columns, loc='center')

fig.tight_layout()

plt.show()


# In[21]:


white_test = het_white(results.resid,  results.model.exog)

labels = ['Тестовая статистика', 'тестовая значиость', 'F-статистика', 'F-тест значимость']

TabWhigt = dict(zip(labels, white_test))
TabWhigt = pd.DataFrame(TabWhigt, index=[0])


# In[22]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=TabWhigt.values, colLabels=TabWhigt.columns, loc='center')

fig.tight_layout()

plt.show()


# In[23]:


#проверка качества модели
X_train, X_test, y_train, y_test = train_test_split(
    nRoomq.to_numpy().reshape(-1, 1), PbSq.to_numpy().reshape(-1, 1),
    test_size=0.3, random_state=0)


# In[24]:


y_test = np.delete(y_test, 133).reshape(-1, 1)
y_train = np.delete(y_train, 133).reshape(-1, 1)
X_test = np.delete(X_test, 133).reshape(-1, 1)
X_train = np.delete(X_train, 133).reshape(-1, 1)


# In[25]:


slr = LinearRegression()


# In[26]:


slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)


# In[27]:


np.where(y_test_pred == min(y_test_pred))


# In[28]:


min(y_test_pred)


# In[29]:


print('MSE train: {:.3f}, test: {:.3f}'.format(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: {:.3f}, test: {:.3f}'.format(
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# In[30]:


plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=65000, xmax=100000, lw=2, color='red')
#plt.xlim([-10, 50])
plt.tight_layout()


# In[31]:


sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(nRoomq.to_numpy().reshape(-1, 1))
y_std = sc_y.fit_transform(PbSq.to_numpy().reshape(-1, 1)).flatten()
# newaxis увеличивает размерность массива, flatten — наооборот

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_std, y_std, test_size=0.3, random_state=0)


# In[32]:


X_train_scaled.std(), X_train_scaled.mean()


# In[33]:


en = ElasticNet(alpha=0.1, l1_ratio=0.5)
en.fit(X_train_scaled, y_train_scaled)
y_train_pred = en.predict(X_train_scaled)
y_test_pred = en.predict(X_test_scaled)
print(en.coef_)

print('MSE train: {:.3f}, test: {:.3f}'.format(
        mean_squared_error(y_train_scaled, y_train_pred),
        mean_squared_error(y_test_scaled, y_test_pred)))
print('R^2 train: {:.3f}, test: {:.3f}'.format(
        r2_score(y_train_scaled, y_train_pred),
        r2_score(y_test_scaled, y_test_pred)))


# In[ ]:


nRomm = np.delete(nRoom.to_numpy(), 5)
Prr = np.delete(Pr.to_numpy(), 5)


# In[34]:


regr = LinearRegression()

quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(nRoomq.to_numpy().reshape(-1, 1))
X_cubic = cubic.fit_transform(nRoomq.to_numpy().reshape(-1, 1))


# In[35]:


X_fit = np.arange(nRoomq.to_numpy().min(), nRoomq.to_numpy().max(), 1)[:, np.newaxis]

regr = regr.fit(nRoomq.to_numpy().reshape(-1, 1), PbSq.to_numpy().reshape(-1, 1))
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(PbSq.to_numpy().reshape(-1, 1), regr.predict(nRoomq.to_numpy().reshape(-1, 1)))


# In[36]:


regr = regr.fit(X_quad, PbSq.to_numpy())
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(PbSq.to_numpy(), regr.predict(X_quad))


# In[37]:


regr = regr.fit(X_cubic, PbSq.to_numpy())
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(PbSq.to_numpy().reshape(-1, 1), regr.predict(X_cubic))


# In[40]:


# отображение результатов
plt.scatter(nRoom, PbS, label='training points', color='lightblue')

plt.plot(X_fit, y_lin_fit, 
         label='линейный (d=1), $R^2={:.2f}$'.format(linear_r2), 
         color='blue', 
         lw=2, 
         linestyle='-')

plt.xlabel('Этаж')
plt.ylabel('Цена за квадрат')
plt.legend(loc='upper right')


# In[ ]:




