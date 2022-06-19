#!/usr/bin/env python
# coding: utf-8

# In[112]:


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


# In[4]:


#Импорт БД
Base = pd.read_csv('C:/Users/Dima/Desktop/АНДАН - workable data (1).csv', delimiter=',', 
                 names=['Имя', 'Цена', 'район', 'площадь', 'Цена за квадрат',
                        'кол-во комнат', 'тип дома', 'этажность', 'этаж', 'срок сдачи'])


# In[5]:


# Функция очистки от выбросов
def hampel(vals_orig):
    vals = vals_orig.copy()    
    difference = np.abs(vals.median()-vals)
    median_abs_deviation = difference.median()
    threshold = 3 * median_abs_deviation
    outlier_idx = difference > threshold
    vals[outlier_idx] = np.nan
    return(vals)


# In[24]:


# Необходимые переменные
Pr     = hampel(Base['Цена'])
Sq     = hampel(Base['площадь'])
nRoom  = hampel(Base['кол-во комнат'])
PbS    = hampel(Base['Цена за квадрат'])
StageH = hampel(Base['этажность'])
StageF = hampel(Base['этаж'])


# In[31]:


#Гипотеза 1:
"""чем больше этажность дома, тем больше этаж на котором находится квартира влияет на стоимость кВ м"""
temp1 = []
temp2 = []

#Поиск кореляции
for i in range(int(np.min(StageH)), int(np.max(StageH))):
    Diff = Base.loc[Base['этажность'] == i]
    
    StageF = Diff.head()[['этаж']].astype(np.float)
    PbS    = Diff.head()[['Цена за квадрат']].astype(np.float)
    
    temp1.append(sp.kendalltau(StageF, PbS)[0])
    temp2.append(sp.spearmanr(StageF, PbS)[0])

temp1 = np.where(np.isnan(temp1), 0., temp1)
temp2 = np.where(np.isnan(temp2), 0., temp2)


# In[32]:


#гистаграмма корреляций по этажности
fig = plt.figure(figsize=(8, 6))
axs = fig.add_subplot(121)

n_bins1 = len(temp1)

axs.set_title('Тау Кендала')

sns.histplot(temp1, kde=True, color='orange')

plt.savefig("гистаграмма корреляций по этажности Тау Кендала.svg")


# In[33]:


fig = plt.figure(figsize=(8, 6))
axs = fig.add_subplot(121)

n_bins2 = len(temp2)

axs.set_title('Корреляция\n Спирмена')

sns.histplot(temp2, kde=True, color='orange')
#fig = sns_plot.get_figure()
plt.savefig("гистаграмма корреляций по этажности Корреляция Спирмена.svg")


# In[34]:


# создание копии БД
Based = Base.copy()
Based['Цена за квадрат'] = hampel(Based['Цена за квадрат'])


# In[35]:


#Категорированная диаграмма Бокса-Уискера
fig, ax = plt.subplots()

sns.boxplot(data=Based, x='этаж', y='Цена за квадрат')

plt.title('Категорированная диаграмма Бокса-Уискера', fontsize=32)
plt.xlabel('этаж', fontsize=32)
plt.ylabel('Цена за квадратный метр', fontsize=32)
ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
fig.set_size_inches(11.7, 8.27)

plt.show()
plt.savefig("Категорированная диаграмма Бокса-Уискера для этажа и цены за квадрат.svg")


# In[36]:


PbSd    = Base['Цена за квадрат']
StageFd = Base['этаж']

Kr = sp.kruskal(StageFd, PbSd)
print(Kr)


# In[37]:


TabBox = {
    'Нулевая гипотеза' : ['Цена за квадратный метр зависит от этажа'],
    'критерий' : ['критерий Краскела-Уоллиса'],
    'значимость' : [Kr[1].round(4)],
    'значение статистики' : [Kr[0].round(4)],
    'решение' : ['Нулевая гипотеза отклонена']
}

Tab = pd.DataFrame(TabBox)


# In[38]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=Tab.values, colLabels=Tab.columns, loc='center')

fig.tight_layout()

plt.show()
plt.savefig("таблица нулевой гипотезы для гипотезы 1.svg")


# In[41]:


#Диаграмма рассеивания для переменных цена за квартиру и площадь квартиры
PbSa    = hampel(Base['Цена за квадрат']) # копии переменных
StageFa = hampel(Base['этаж'])

fig, ax = plt.subplots()

ax.scatter(StageFa, PbSa, c = 'deeppink')    

ax.set_title('Диаграмма рассеивания для переменных цена за квартиру и этажа', fontsize=32)
plt.xlabel('этаж', fontsize=32)
plt.ylabel('Цена за квадратный метр', fontsize=32)

fig.set_figwidth(11.7 * 2)    
fig.set_figheight(8.27 * 2)   

plt.show()
plt.savefig("Диаграмма рассеивания для переменных цена за квартиру и этажа.svg")


# In[67]:


PbSq    = Base['Цена за квадрат']
StageFq = Base['этаж']

slr = LinearRegression()

slr.fit(StageFq.to_numpy().reshape(-1, 1), PbSq.to_numpy().reshape(-1, 1))

y_pred = slr.predict(StageFq.to_numpy().reshape(-1, 1))

print(slr.coef_[0])
print(slr.intercept_)


# In[141]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

results = smf.ols('StageFq ~ PbSq', data=Base).fit()
print(results.summary())


# In[190]:


results.conf_int()[0]


# In[195]:


TabStat = {
    'переменная' : ['StageFq', 'PbS'],
    'Коэф.' : [results.params[0].round(3), results.params[1].round(3)],
    'станд. ошибка' : [results.bse[0].round(3), results.bse[1].round(3)],
    't-статистика' : [results.tvalues[0].round(3), results.tvalues[1].round(3)],
    'p-уровень' : [results.pvalues[0].round(3), results.pvalues[1].round(3)],
    '95% дов интервал левый' : [results.conf_int()[0][0].round(3), results.conf_int()[0][1].round(3)],
    '95% дов интервал правый' : [results.conf_int()[1][0].round(3), results.conf_int()[1][1].round(3)]
}

TabStat = pd.DataFrame(TabStat)
print(TabStat)


# In[197]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=TabStat.values, colLabels=TabStat.columns, loc='center')

fig.tight_layout()

plt.show()


# In[74]:


#проверка качества модели
X_train, X_test, y_train, y_test = train_test_split(
    StageFq.to_numpy().reshape(-1, 1), PbSq.to_numpy().reshape(-1, 1),
    test_size=0.3, random_state=0)


# In[75]:


slr = LinearRegression()


# In[76]:


slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)


# In[78]:


print('MSE train: {:.3f}, test: {:.3f}'.format(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: {:.3f}, test: {:.3f}'.format(
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# In[98]:


plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=85000, xmax=102000, lw=2, color='red')
#plt.xlim([-10, 50])
plt.tight_layout()


# In[113]:


sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(StageFq.to_numpy().reshape(-1, 1))
y_std = sc_y.fit_transform(PbSq.to_numpy().reshape(-1, 1)).flatten()
# newaxis увеличивает размерность массива, flatten — наооборот
# https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#numpy.newaxis
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.flatten.html

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_std, y_std, test_size=0.3, random_state=0)


# In[114]:


X_train_scaled.std(), X_train_scaled.mean()


# In[115]:


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


# In[125]:


regr = LinearRegression()

quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(StageFq.to_numpy().reshape(-1, 1))
X_cubic = cubic.fit_transform(StageFq.to_numpy().reshape(-1, 1))


# In[126]:


X_fit = np.arange(StageFq.to_numpy().min(), StageFq.to_numpy().max(), 1)[:, np.newaxis]

regr = regr.fit(StageFq.to_numpy().reshape(-1, 1), PbSq.to_numpy().reshape(-1, 1))
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(PbSq.to_numpy().reshape(-1, 1), regr.predict(StageFq.to_numpy().reshape(-1, 1)))


# In[127]:


regr = regr.fit(X_quad, PbSq.to_numpy())
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(PbSq.to_numpy(), regr.predict(X_quad))


# In[128]:


regr = regr.fit(X_cubic, PbSq.to_numpy())
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(PbSq.to_numpy().reshape(-1, 1), regr.predict(X_cubic))


# In[131]:


# отображение результатов
plt.scatter(StageFq, PbSq, label='training points', color='lightgray')

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




