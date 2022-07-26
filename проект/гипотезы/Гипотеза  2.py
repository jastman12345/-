#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


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


# Необходимые переменные
Pr     = hampel(Base['Цена'])
Sq     = hampel(Base['площадь'])
nRoom  = hampel(Base['кол-во комнат'])
PbS    = hampel(Base['Цена за квадрат'])
StageH = hampel(Base['этажность'])
StageF = hampel(Base['этаж'])


# In[6]:


#Гипотеза 2:
'''Площадь квартиры наибольшим образом влияет на ее стоимость, 
   однако если квартира расположена в центре города площадь 
   квартиры может уменьшаться, а стоимость возрастать'''

Sqm     = Base['площадь']
PbSm    = Base['Цена за квадрат']

# Корреляция площади Квартиры и стоимости квалратного метра
PIR = sp.pearsonr(Sqm, PbSm)
KT = sp.kendalltau(Sqm, PbSm)
SP = sp.spearmanr(Sqm, PbSm)


# In[7]:


Tab2 = {
    'пирсона' : PIR,
    'Спирмена' : SP,
    'Тау Кендалла' : KT
}

Tab = pd.DataFrame(Tab2)


# In[8]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=Tab.values, colLabels=Tab.columns, loc='center')

fig.tight_layout()

plt.show()
plt.savefig("таблица корреляций для гипотезы 2.svg")


# In[9]:


Based = Base.copy()
Based['Цена за квадрат'] = hampel(Base['Цена за квадрат'])


# In[14]:


#Распределенеи цены за квадрат
fig = plt.figure(figsize=(8, 6))
axs = fig.add_subplot(121)

n_bins1 = 10
X = Based['Цена за квадрат']

sns.histplot(X, kde=True)
plt.savefig("Распределенеи цены за квадрат.svg")


# In[16]:


Basedd = Base.copy()
Basedd['Цена за квадрат'] = hampel(Base['Цена за квадрат'])
Basedd['площадь'] = hampel(Base['площадь'])


# In[17]:


#Категорированная диаграмма Бокса-Уискера
fig, ax = plt.subplots()

sns.boxplot(data=Basedd, x='площадь', y='Цена за квадрат')

plt.title('Категорированная диаграмма Бокса-Уискера', fontsize=32)
plt.xlabel('площадь', fontsize=32)
plt.ylabel('Цена за квадратный метр', fontsize=32)
plt.tick_params(axis='x', rotation=70)
ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
fig.set_size_inches(11.7 * 2, 8.27 * 2)

plt.show()
plt.savefig("Категорированная диаграмма Бокса-Уискера площадь и цена.svg")


# In[18]:


Sqf     = Base['площадь']
PbSf    = Base['Цена за квадрат']

Kr = sp.kruskal(Sqf, PbSf)
print(Kr)


# In[19]:


TabBox = {
    'Нулевая гипотеза' : ['Цена за квадратный метр зависит от площади'],
    'критерий' : ['критерий Краскела-Уоллиса'],
    'значимость' : [Kr[1]],
    'значение статистики' : [Kr[0]],
    'решение' : ['Нулевая гипотеза откланена']
}

Tab = pd.DataFrame(TabBox)


# In[20]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=Tab.values, colLabels=Tab.columns, loc='center')

fig.tight_layout()

plt.show()
plt.savefig("таблица нулевой гипотезы для гипотезы 2.svg")


# In[21]:


#отбор данных для каждого района
DS1 = Base.loc[Base['район'] == 'р-н Центральный']
DS2 = Base.loc[Base['район'] == 'р-н Привокзальный']
DS3 = Base.loc[Base['район'] == 'р-н Советский']
DS4 = Base.loc[Base['район'] == 'р-н Зареченский']
DS5 = Base.loc[Base['район'] == 'р-н Пролетарский']


# In[22]:


#оздание массивов с средней ценной и площадью квартир от района
SqR = np.array([DS1['площадь'].mean(), DS2['площадь'].mean(), DS3['площадь'].mean(), DS4['площадь'].mean(), 
                DS5['площадь'].mean()])
PbSR = np.array([DS1['Цена за квадрат'].mean(), DS2['Цена за квадрат'].mean(), 
        DS3['Цена за квадрат'].mean(), DS4['Цена за квадрат'].mean(), DS5['Цена за квадрат'].mean()])
I = np.array(['р-н Центральный', 'р-н Привокзальный', 'р-н Советский','Зареченский', 'Пролетарский',])


# In[23]:


#строим график
fig, axd = plt.subplot_mosaic([['upleft', 'right']])

axd['upleft'].plot(I, SqR)
axd['right'].plot(I, PbSR)

axd['upleft'].tick_params(axis='x', rotation=70)
axd['right'].tick_params(axis='x', rotation=70)
fig.set_size_inches(11.7, 8.27)

axd['upleft'].set_title('Площадь квартиры\n от района')
axd['right'].set_title('Цена за квадрат\nный метр');

plt.subplots_adjust(wspace=0.5)
plt.savefig("цена и площадь от района.svg")


# In[24]:


MegaBased = Base.copy()
MegaBased['площадь'] = hampel(MegaBased['площадь'])
MegaBased['Цена за квадрат'] = hampel(MegaBased['Цена за квадрат'])


# In[28]:


#Диаграмма рассеивания для переменных цена за квартиру и площадь квартиры
fig, ax = plt.subplots()

data = MegaBased['площадь']
y = MegaBased['Цена за квадрат']

ax.scatter(data, y, c = 'darkblue')    

ax.set_title('Диаграмма рассеивания для переменных цена за квартиру и площадь квартиры', fontsize=32)
plt.xlabel('площадь', fontsize=32)
plt.ylabel('Цена за квадратный метр', fontsize=32)

fig.set_figwidth(11.7 * 2)    
fig.set_figheight(8.27 * 2)   

plt.show()
plt.savefig("Диаграмма рассеивания для переменных цена за квартиру и площадь квартиры.svg")


# In[29]:


Basec = Base[Base['площадь'] < 5000]


# In[30]:


Sq     = Basec['площадь']
PbS    = Basec['Цена за квадрат']

#регрессионная модель
slr = LinearRegression()

slr.fit(Sq.to_numpy().reshape(-1, 1), PbS.to_numpy().reshape(-1, 1))

y_pred = slr.predict(Sq.to_numpy().reshape(-1, 1))

print(slr.coef_[0])
print(slr.intercept_)


# In[31]:


print(Sq)


# In[32]:


import statsmodels.formula.api as smf

results = smf.ols('PbS ~ Sq', data=Basec).fit()
print(results.summary())


# In[33]:


TabStat = {
    'переменная' : ['Sq'],
    'Коэф.' : [results.params[1].round(3)],
    'станд. ошибка' : [results.bse[1].round(3)],
    't-статистика' : [results.tvalues[1].round(3)],
    'p-уровень' : [results.pvalues[1].round(3)],
    '95% дов интервал левый' : [results.conf_int()[0][1].round(3)],
    '95% дов интервал правый' : [results.conf_int()[1][1].round(3)]
}

TabStat = pd.DataFrame(TabStat)
print(TabStat)


# In[34]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=TabStat.values, colLabels=TabStat.columns, loc='center')

fig.tight_layout()

plt.show()


# In[35]:


white_test = het_white(results.resid,  results.model.exog)

labels = ['Тестовая статистика', 'тестовая значиость', 'F-статистика', 'F-тест значимость']

TabWhigt = dict(zip(labels, white_test))
TabWhigt = pd.DataFrame(TabWhigt, index=[0])


# In[36]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=TabWhigt.values, colLabels=TabWhigt.columns, loc='center')

fig.tight_layout()

plt.show()


# In[37]:


#проверка качества модели
X_train, X_test, y_train, y_test = train_test_split(
    Sq.to_numpy().reshape(-1, 1), PbS.to_numpy().reshape(-1, 1),
    test_size=0.3, random_state=0)


# In[38]:


slr = LinearRegression()


# In[39]:


slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)


# In[40]:


print('MSE train: {:.3f}, test: {:.3f}'.format(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: {:.3f}, test: {:.3f}'.format(
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# In[41]:


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


# In[42]:


sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(Sq.to_numpy().reshape(-1, 1))
y_std = sc_y.fit_transform(PbS.to_numpy().reshape(-1, 1)).flatten()
# newaxis увеличивает размерность массива, flatten — наооборот

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_std, y_std, test_size=0.3, random_state=0)


# In[43]:


X_train_scaled.std(), X_train_scaled.mean()


# In[49]:


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


# In[50]:


regr = LinearRegression()

quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(Sq.to_numpy().reshape(-1, 1))
X_cubic = cubic.fit_transform(Sq.to_numpy().reshape(-1, 1))


# In[51]:


X_fit = np.arange(Sq.to_numpy().min(), Sq.to_numpy().max(), 1)[:, np.newaxis]

regr = regr.fit(Sq.to_numpy().reshape(-1, 1), PbS.to_numpy().reshape(-1, 1))
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(PbS.to_numpy().reshape(-1, 1), regr.predict(Sq.to_numpy().reshape(-1, 1)))


# In[52]:


regr = regr.fit(X_quad, PbS.to_numpy())
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(PbS.to_numpy(), regr.predict(X_quad))


# In[53]:


regr = regr.fit(X_cubic, PbS.to_numpy())
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(PbS.to_numpy().reshape(-1, 1), regr.predict(X_cubic))


# In[55]:


# отображение результатов
plt.scatter(Sq, PbS, label='training points', color='lightblue')

plt.plot(X_fit, y_lin_fit, 
         label='линейный (d=1), $R^2={:.2f}$'.format(linear_r2), 
         color='blue', 
         lw=2, 
         linestyle='-')

plt.xlabel('Этаж')
plt.ylabel('Цена за квадрат')
plt.legend(loc='upper right')

