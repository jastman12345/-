#!/usr/bin/env python
# coding: utf-8

# In[33]:


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


# In[2]:


#Импорт БД
Base = pd.read_csv('C:/Users/Dima/Desktop/АНДАН - workable data (1).csv', delimiter=',', 
                 names=['Имя', 'Цена', 'район', 'площадь', 'Цена за квадрат',
                        'кол-во комнат', 'тип дома', 'этажность', 'этаж', 'срок сдачи'])


# In[96]:


# Функция очистки от выбросов
def hampel(vals_orig):
    vals = vals_orig.copy()    
    difference = np.abs(vals.median()-vals)
    median_abs_deviation = difference.median()
    threshold = 3 * median_abs_deviation
    outlier_idx = difference > threshold
    vals[outlier_idx] = np.nan
    return(vals)


# In[97]:


# Необходимые переменные
Pr     = hampel(Base['Цена'])
Sq     = hampel(Base['площадь'])
nRoom  = hampel(Base['кол-во комнат'])
PbS    = hampel(Base['Цена за квадрат'])
StageH = hampel(Base['этажность'])
StageF = hampel(Base['этаж'])


# In[5]:


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


# In[6]:


Tab2 = {
    'пирсона' : PIR,
    'Спирмена' : SP,
    'Тау Кендалла' : KT
}

Tab = pd.DataFrame(Tab2)


# In[7]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=Tab.values, colLabels=Tab.columns, loc='center')

fig.tight_layout()

plt.show()
plt.savefig("таблица корреляций для гипотезы 2.svg")


# In[10]:


Based = Base.copy()
Based['Цена за квадрат'] = hampel(Base['Цена за квадрат'])


# In[11]:


#Распределенеи цены за квадрат
fig = plt.figure(figsize=(8, 6))
axs = fig.add_subplot(121)

n_bins1 = 10
X = Based['Цена за квадрат']

axs.hist(X, bins=n_bins1)
axs.set_title('Распределенеи цены\n за квадрат')

sns.histplot(X, kde=True, color='red')
plt.savefig("Распределенеи цены за квадрат.svg")


# In[12]:


Basedd = Base.copy()
Basedd['Цена за квадрат'] = hampel(Base['Цена за квадрат'])
Basedd['площадь'] = hampel(Base['площадь'])


# In[13]:


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


# In[14]:


Sqf     = Base['площадь']
PbSf    = Base['Цена за квадрат']

Kr = sp.kruskal(Sqf, PbSf)
print(Kr)


# In[15]:


TabBox = {
    'Нулевая гипотеза' : ['Цена за квадратный метр зависит от площади'],
    'критерий' : ['критерий Краскела-Уоллиса'],
    'значимость' : [Kr[1]],
    'значение статистики' : [Kr[0]],
    'решение' : ['Нулевая гипотеза откланена']
}

Tab = pd.DataFrame(TabBox)


# In[16]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=Tab.values, colLabels=Tab.columns, loc='center')

fig.tight_layout()

plt.show()
plt.savefig("таблица нулевой гипотезы для гипотезы 2.svg")


# In[17]:


#отбор данных для каждого района
DS1 = Base.loc[Base['район'] == 'р-н Центральный']
DS2 = Base.loc[Base['район'] == 'р-н Привокзальный']
DS3 = Base.loc[Base['район'] == 'р-н Советский']
DS4 = Base.loc[Base['район'] == 'р-н Зареченский']
DS5 = Base.loc[Base['район'] == 'р-н Пролетарский']


# In[18]:


#оздание массивов с средней ценной и площадью квартир от района
SqR = np.array([DS1['площадь'].mean(), DS2['площадь'].mean(), DS3['площадь'].mean(), DS4['площадь'].mean(), 
                DS5['площадь'].mean()])
PbSR = np.array([DS1['Цена за квадрат'].mean(), DS2['Цена за квадрат'].mean(), 
        DS3['Цена за квадрат'].mean(), DS4['Цена за квадрат'].mean(), DS5['Цена за квадрат'].mean()])
I = np.array(['р-н Центральный', 'р-н Привокзальный', 'р-н Советский','Зареченский', 'Пролетарский',])


# In[19]:


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


# In[21]:


MegaBased = Base.copy()
MegaBased['площадь'] = hampel(MegaBased['площадь'])
MegaBased['Цена за квадрат'] = hampel(MegaBased['Цена за квадрат'])


# In[23]:


#Диаграмма рассеивания для переменных цена за квартиру и площадь квартиры
fig, ax = plt.subplots()

data = MegaBased['площадь']
y = MegaBased['Цена за квадрат']

ax.scatter(data, y, c = 'deeppink')    

ax.set_title('Диаграмма рассеивания для переменных цена за квартиру и площадь квартиры', fontsize=32)
plt.xlabel('площадь', fontsize=32)
plt.ylabel('Цена за квадратный метр', fontsize=32)

fig.set_figwidth(11.7 * 2)    
fig.set_figheight(8.27 * 2)   

plt.show()
plt.savefig("Диаграмма рассеивания для переменных цена за квартиру и площадь квартиры.svg")


# In[209]:


Basec = Base[Base['площадь'] < 5000]


# In[210]:


Sq     = Basec['площадь']
PbS    = Basec['Цена за квадрат']

#регрессионная модель
slr = LinearRegression()

slr.fit(Sq.to_numpy().reshape(-1, 1), PbS.to_numpy().reshape(-1, 1))

y_pred = slr.predict(Sq.to_numpy().reshape(-1, 1))

print(slr.coef_[0])
print(slr.intercept_)


# In[211]:


print(Sq)


# In[212]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

results = smf.ols('Sq ~ PbS', data=Basec).fit()
print(results.summary())


# In[213]:


TabStat = {
    'переменная' : ['Sq', 'PbS'],
    'Коэф.' : [results.params[0].round(3), results.params[1].round(3)],
    'станд. ошибка' : [results.bse[0].round(3), results.bse[1].round(3)],
    't-статистика' : [results.tvalues[0].round(3), results.tvalues[1].round(3)],
    'p-уровень' : [results.pvalues[0].round(3), results.pvalues[1].round(3)],
    '95% дов интервал левый' : [results.conf_int()[0][0].round(3), results.conf_int()[0][1].round(3)],
    '95% дов интервал правый' : [results.conf_int()[1][0].round(3), results.conf_int()[1][1].round(3)]
}

TabStat = pd.DataFrame(TabStat)
print(TabStat)


# In[214]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=TabStat.values, colLabels=TabStat.columns, loc='center')

fig.tight_layout()

plt.show()


# In[ ]:





# In[215]:


#проверка качества модели
X_train, X_test, y_train, y_test = train_test_split(
    Sq.to_numpy().reshape(-1, 1), PbS.to_numpy().reshape(-1, 1),
    test_size=0.3, random_state=0)


# In[216]:


slr = LinearRegression()


# In[217]:


slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)


# In[218]:


print('MSE train: {:.3f}, test: {:.3f}'.format(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: {:.3f}, test: {:.3f}'.format(
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# In[222]:


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


# In[223]:


sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(Sq.to_numpy().reshape(-1, 1))
y_std = sc_y.fit_transform(PbS.to_numpy().reshape(-1, 1)).flatten()
# newaxis увеличивает размерность массива, flatten — наооборот

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_std, y_std, test_size=0.3, random_state=0)


# In[224]:


X_train_scaled.std(), X_train_scaled.mean()


# In[225]:


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


# In[226]:


regr = LinearRegression()

quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(Sq.to_numpy().reshape(-1, 1))
X_cubic = cubic.fit_transform(Sq.to_numpy().reshape(-1, 1))


# In[227]:


X_fit = np.arange(Sq.to_numpy().min(), Sq.to_numpy().max(), 1)[:, np.newaxis]

regr = regr.fit(Sq.to_numpy().reshape(-1, 1), PbS.to_numpy().reshape(-1, 1))
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(PbS.to_numpy().reshape(-1, 1), regr.predict(Sq.to_numpy().reshape(-1, 1)))


# In[228]:


regr = regr.fit(X_quad, PbS.to_numpy())
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(PbS.to_numpy(), regr.predict(X_quad))


# In[229]:


regr = regr.fit(X_cubic, PbS.to_numpy())
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(PbS.to_numpy().reshape(-1, 1), regr.predict(X_cubic))


# In[230]:


# отображение результатов
plt.scatter(Sq, PbS, label='training points', color='lightgray')

plt.plot(X_fit, y_lin_fit, 
         label='линейный (d=1), $R^2={:.2f}$'.format(linear_r2), 
         color='blue', 
         lw=2, 
         linestyle=':')

plt.plot(X_fit, y_quad_fit, 
         label='квадратичный (d=2), $R^2={:.2f}$'.format(quadratic_r2),
         color='red', 
         lw=2,
         linestyle='-')

plt.plot(X_fit, y_cubic_fit, 
         label='кубический (d=3), $R^2={:.2f}$'.format(cubic_r2),
         color='green', 
         lw=2, 
         linestyle='--')

plt.xlabel('Этаж')
plt.ylabel('Цена за квадрат')
plt.legend(loc='upper right')


# In[ ]:




