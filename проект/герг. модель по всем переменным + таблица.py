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
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib as mpl
from statsmodels.graphics.regressionplots import influence_plot


# In[3]:


#Импорт БД
Base = pd.read_csv('C:/Users/Dima/Desktop/АНДАН - workable data (1).csv', delimiter=',', 
                 names=['Имя', 'Цена', 'район', 'площадь', 'Цена за квадрат',
                        'кол-во комнат', 'тип дома', 'этажность', 'этаж', 'срок сдачи'])


# In[4]:


# Функция очистки от выбросов
def func(df):
    vals = df.copy()    
    Q1 =  vals.quantile(0.25)
    Q3 = vals.quantile(0.75)
    IQR = Q3 - Q1
    vals = vals[(vals > Q1-1.5*IQR ) | (vals < Q1+1.5*IQR)]
    return vals


# In[5]:


Pr     = (func(Base['Цена']))
Sq     = func(Base['площадь'])
nRoom  = func(Base['кол-во комнат'])
PbS    = func(Base['Цена за квадрат'])
StageH = func(Base['этажность'])
StageF = func(Base['этаж'])


# In[6]:


max(Pr)


# In[7]:


model = ['район', 'площадь', 'Цена за квадрат','кол-во комнат', 'тип дома', 'этажность', 'этаж', 'срок сдачи']


# In[63]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

results = smf.ols('Pr ~ Sq + nRoom + PbS + StageH + StageF', data=Base).fit()
print(results.summary())


# In[64]:


results.rsquared


# In[65]:


TabStat = {
    'переменная' : ['coff','Sq','nRoom', 'PbS', 'StageH', 'StageF'],
    'Коэф.' : [results.params[0].round(3), results.params[1].round(3), 
               results.params[2].round(3),
               results.params[3].round(3), results.params[4].round(3),
               results.params[5].round(3)
              ],
    'станд. ошибка' : [results.bse[0].round(3), results.bse[1].round(3), 
                       results.bse[2].round(3),
                       results.bse[3].round(3), results.bse[4].round(3),
                       results.bse[5].round(3)
                      ],
    't-статистика' : [results.tvalues[0].round(3), results.tvalues[1].round(3), 
                      results.tvalues[2].round(3),
                      results.tvalues[3].round(3), results.tvalues[4].round(3),
                      results.tvalues[5].round(3)
                     ],
    'p-уровень' : [results.pvalues[0].round(3), results.pvalues[1].round(3), 
                   results.pvalues[2].round(3),
                   results.pvalues[3].round(3), results.pvalues[4].round(3),
                   results.pvalues[5].round(3)
                  ],
    '95% дов интервал левый' : [results.conf_int()[0][0].round(3), results.conf_int()[0][1].round(3), 
                                results.conf_int()[0][2].round(3),
                                results.conf_int()[0][3].round(3), results.conf_int()[0][4].round(3),
                                results.conf_int()[0][5].round(3)
                               ],
    '95% дов интервал правый' : [results.conf_int()[1][0].round(3), results.conf_int()[1][1].round(3), 
                                 results.conf_int()[1][2].round(3),
                                 results.conf_int()[1][3].round(3), results.conf_int()[1][4].round(3),
                                 results.conf_int()[1][5].round(3)
                                ]
}

TabStat = pd.DataFrame(TabStat)
print(TabStat)


# In[66]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=TabStat.values, colLabels=TabStat.columns, loc='center')

fig.tight_layout()

plt.show()


# In[70]:


TabMOD = {
    '$R^2$.' : [results.rsquared.round(3)],
    '$Adj_R^2$' : [results.rsquared_adj.round(3)],
    'AIC' : [results.aic.round(3)],
}

TabMOD = pd.DataFrame(TabMOD)
print(TabMOD)


# In[71]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=TabMOD.values, colLabels=TabMOD.columns, loc='center')

fig.tight_layout()

plt.show()


# In[13]:


regr = LinearRegression()

quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(Sq.to_numpy().reshape(-1, 1))
X_cubic = cubic.fit_transform(Sq.to_numpy().reshape(-1, 1))


# In[14]:


X_fit = np.arange(Sq.to_numpy().min(), Sq.to_numpy().max(), 1)[:, np.newaxis]

regr = regr.fit(Sq.to_numpy().reshape(-1, 1), Pr.to_numpy().reshape(-1, 1))
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(Pr.to_numpy().reshape(-1, 1), regr.predict(Sq.to_numpy().reshape(-1, 1)))


# In[15]:


nRomm = np.delete(nRoom.to_numpy(), 5)
Prr = np.delete(Pr.to_numpy(), 5)


# In[16]:


regr1 = LinearRegression()

X_quad1 = quadratic.fit_transform(nRomm.reshape(-1, 1))
X_cubic1 = cubic.fit_transform(nRomm.reshape(-1, 1))


# In[17]:


X_fit1 = np.arange(nRomm.min(), nRomm.max(), 1)[:, np.newaxis]

regr1 = regr1.fit(nRomm.reshape(-1, 1), Prr.reshape(-1, 1))
y_lin_fit1 = regr1.predict(X_fit1)
linear_r21 = r2_score(Prr.reshape(-1, 1), regr1.predict(nRomm.reshape(-1, 1)))


# In[18]:


regr2 = LinearRegression()

X_quad2 = quadratic.fit_transform(PbS.to_numpy().reshape(-1, 1))
X_cubic2 = cubic.fit_transform(PbS.to_numpy().reshape(-1, 1))


# In[19]:


X_fit2 = np.arange(PbS.to_numpy().min(), PbS.to_numpy().max(), 1)[:, np.newaxis]

regr2 = regr2.fit(PbS.to_numpy().reshape(-1, 1), Pr.to_numpy().reshape(-1, 1))
y_lin_fit2 = regr2.predict(X_fit2)
linear_r22 = r2_score(Pr.to_numpy().reshape(-1, 1), regr2.predict(PbS.to_numpy().reshape(-1, 1)))


# In[20]:


regr3 = LinearRegression()

quadratic3 = PolynomialFeatures(degree=2)
cubic3 = PolynomialFeatures(degree=3)
X_quad3 = quadratic.fit_transform(StageH.to_numpy().reshape(-1, 1))
X_cubic3 = cubic.fit_transform(StageH.to_numpy().reshape(-1, 1))


# In[21]:


X_fit3 = np.arange(StageH.to_numpy().min(), StageH.to_numpy().max(), 1)[:, np.newaxis]

regr3 = regr3.fit(StageH.to_numpy().reshape(-1, 1), Pr.to_numpy().reshape(-1, 1))
y_lin_fit3 = regr3.predict(X_fit3)
linear_r23 = r2_score(Pr.to_numpy().reshape(-1, 1), regr3.predict(StageH.to_numpy().reshape(-1, 1)))


# In[22]:


regr4 = LinearRegression()

quadratic4 = PolynomialFeatures(degree=2)
cubic4 = PolynomialFeatures(degree=3)
X_quad4 = quadratic.fit_transform(StageF.to_numpy().reshape(-1, 1))
X_cubic4 = cubic.fit_transform(StageF.to_numpy().reshape(-1, 1))


# In[23]:


X_fit4 = np.arange(StageF.to_numpy().min(), StageF.to_numpy().max(), 1)[:, np.newaxis]

regr4 = regr4.fit(StageF.to_numpy().reshape(-1, 1), Pr.to_numpy().reshape(-1, 1))
y_lin_fit4 = regr4.predict(X_fit4)
linear_r24 = r2_score(Pr.to_numpy().reshape(-1, 1), regr4.predict(StageF.to_numpy().reshape(-1, 1)))


# In[24]:


plt.figure(figsize=(16,10), dpi= 80)

plt.subplot(2, 3, 1)
plt.scatter(Sq, Pr, label='training points', color='lightblue')
plt.plot(X_fit, y_lin_fit, 
         label='линейный (d=1), $R^2={:.2f}$'.format(linear_r2), 
         color='blue', 
         lw=2, 
         linestyle='-')

plt.title("Площадь-Цена")
plt.legend(loc='upper right')


plt.subplot(2, 3, 2)
plt.scatter(nRomm, Prr, label='training points', color='lightblue')
plt.plot(X_fit1, y_lin_fit1, 
         label='линейный (d=1), $R^2={:.2f}$'.format(linear_r2), 
         color='blue', 
         lw=2, 
         linestyle='-')

plt.title("кол-во комнат-Цена")
plt.legend(loc='upper right')


plt.subplot(2, 3, 4)
plt.scatter(PbS, Pr, label='training points', color='lightblue')
plt.plot(X_fit2, y_lin_fit2, 
         label='линейный (d=1), $R^2={:.2f}$'.format(linear_r2), 
         color='blue', 
         lw=2, 
         linestyle='-')

plt.title("Цена за квадрат-Цена")
plt.legend(loc='upper right')


plt.subplot(2, 3, 5)
plt.scatter(StageH, Pr, label='training points', color='lightblue')
plt.plot(X_fit3, y_lin_fit3, 
         label='линейный (d=1), $R^2={:.2f}$'.format(linear_r2), 
         color='blue', 
         lw=2, 
         linestyle='-')

plt.title("этажность-Цена")
plt.legend(loc='upper right')


plt.subplot(1, 3, 3)
plt.scatter(StageF, Pr, label='training points', color='lightblue')
plt.plot(X_fit4, y_lin_fit4, 
         label='линейный (d=1), $R^2={:.2f}$'.format(linear_r2), 
         color='blue', 
         lw=2, 
         linestyle='-')

plt.title("этаж-Цена")
plt.legend(loc='upper right')

plt.show()


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(
    Sq.to_numpy().reshape(-1, 1), Pr.to_numpy().reshape(-1, 1),
    test_size=0.3, random_state=0)

results = smf.ols('Pr ~ Sq', data=Base).fit()
model = sm.OLS(y_train, sm.add_constant(X_train)).fit()


# In[26]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(
    nRoom.to_numpy().reshape(-1, 1), Pr.to_numpy().reshape(-1, 1),
    test_size=0.3, random_state=0)

results1 = smf.ols('Pr ~ nRoom', data=Base).fit()
model1 = sm.OLS(y_train, sm.add_constant(X_train1)).fit()


# In[27]:


X_train2, X_test, y_train2, y_test2 = train_test_split(
    PbS.to_numpy().reshape(-1, 1), Pr.to_numpy().reshape(-1, 1),
    test_size=0.3, random_state=0)

results2 = smf.ols('Pr ~ PbS', data=Base).fit()
model2 = sm.OLS(y_train, sm.add_constant(X_train2)).fit()


# In[28]:


X_train3, X_test3, y_train3, y_test3 = train_test_split(
    StageH.to_numpy().reshape(-1, 1), Pr.to_numpy().reshape(-1, 1),
    test_size=0.3, random_state=0)

results3 = smf.ols('Pr ~ StageH', data=Base).fit()
model3 = sm.OLS(y_train, sm.add_constant(X_train3)).fit()


# In[29]:


X_train4, X_test4, y_train4, y_tes4t = train_test_split(
    StageF.to_numpy().reshape(-1, 1), Pr.to_numpy().reshape(-1, 1),
    test_size=0.3, random_state=0)

results4 = smf.ols('Pr ~ StageF', data=Base).fit()
model4 = sm.OLS(y_train, sm.add_constant(X_train4)).fit()


# In[30]:


influence_plot(results, external=True, alpha=1e-10, criterion='cooks', size=30, plot_alpha=0.25)

plt.show()


# In[31]:


influence_plot(results1, external=True, alpha=1e-10, criterion='cooks', size=30, plot_alpha=0.25)

plt.title("кол-во комнат-Цена")
plt.show()


# In[32]:


influence_plot(results3, external=True, alpha=1e-10, criterion='cooks', size=30, plot_alpha=0.25)

plt.title("этажность-Цена")
plt.show()


# In[33]:


influence_plot(results4, external=True, alpha=1e-10, criterion='cooks', size=30, plot_alpha=0.25)

plt.title("этаж-Цена")

plt.show()


# In[34]:


influence_plot(results2, external=True, alpha=1e-10, criterion='cooks', size=30, plot_alpha=0.25)

plt.title("Цена за квадрат-Цена")
plt.show()


# In[38]:


# удаление выбросов
Prq     = np.delete(Pr.to_numpy(), [410, 14, 5, 0, 1, 15, 24, 25])
Sqq     = np.delete(Sq.to_numpy(), [410, 14, 5, 0, 1, 15, 24, 25])
nRoomq  = np.delete(nRoom.to_numpy(), [410, 14, 5, 0, 1, 15, 24, 25])
PbSq    = np.delete(PbS.to_numpy(), [410, 14, 5, 0, 1, 15, 24, 25])
StageHq = np.delete(StageH.to_numpy(), [410, 14, 5, 0, 1, 15, 24, 25])
StageFq = np.delete(StageF.to_numpy(), [410, 14, 5, 0, 1, 15, 24, 25])


# In[39]:


md = pd.DataFrame({'Pr' : Prq, 'Sq' : Sqq, 'nRoom' : nRoomq,
                   'PbS' : PbSq, 'StageH' : StageHq, 'StageF' : StageFq})
print(md)


# In[40]:


#построение финальной модели
results = smf.ols('Pr ~ Sq + nRoom + PbS + StageH + StageF', data=md).fit()
print(results.summary())


# In[41]:


TabStat = {
    'переменная' : ['Sq','nRoom', 'PbS', 'StageH', 'StageF'],
    'Коэф.' : [results.params[1].round(3), results.params[2].round(3),
               results.params[3].round(3), results.params[4].round(3),
               results.params[5].round(3)
              ],
    'станд. ошибка' : [results.bse[1].round(3), results.bse[2].round(3),
                       results.bse[3].round(3), results.bse[4].round(3),
                       results.bse[5].round(3)
                      ],
    't-статистика' : [results.tvalues[1].round(3), results.tvalues[2].round(3),
                      results.tvalues[3].round(3), results.tvalues[4].round(3),
                      results.tvalues[5].round(3)
                     ],
    'p-уровень' : [results.pvalues[1].round(3), results.pvalues[2].round(3),
                   results.pvalues[3].round(3), results.pvalues[4].round(3),
                   results.pvalues[5].round(3)
                  ],
    '95% дов интервал левый' : [results.conf_int()[0][1].round(3), results.conf_int()[0][2].round(3),
                                results.conf_int()[0][3].round(3), results.conf_int()[0][4].round(3),
                                results.conf_int()[0][5].round(3)
                               ],
    '95% дов интервал правый' : [results.conf_int()[1][1].round(3), results.conf_int()[1][2].round(3),
                                 results.conf_int()[1][3].round(3), results.conf_int()[1][4].round(3),
                                 results.conf_int()[1][5].round(3)
                                ]
}

TabStat = pd.DataFrame(TabStat)
print(TabStat)


# In[42]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=TabStat.values, colLabels=TabStat.columns, loc='center')

fig.tight_layout()

plt.show()


# In[58]:


TabMOD = {
    '$R^2$.' : [results.rsquared.round(3)],
    '$Adj_R^2$' : [results.rsquared_adj.round(3)],
    'AIC' : [results.aic.round(3)],
}

TabMOD = pd.DataFrame(TabMOD)
print(TabMOD)


# In[59]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=TabMOD.values, colLabels=TabMOD.columns, loc='center')

fig.tight_layout()

plt.show()


# In[45]:


white_test = het_white(results.resid,  results.model.exog)

labels = ['Тестовая статистика', 'тестовая значимость', 'F-статистика', 'F-тест значимость']

TabWhigt = dict(zip(labels, white_test))
TabWhigt = pd.DataFrame(TabWhigt, index=[0])


# In[46]:


fig, ax = plt.subplots(figsize=(15, 13))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=TabWhigt.values, colLabels=TabWhigt.columns, loc='center')

fig.tight_layout()

plt.show()


# In[47]:


regr = LinearRegression()

quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(Sqq.reshape(-1, 1))
X_cubic = cubic.fit_transform(Sqq.reshape(-1, 1))


# In[48]:


X_fit = np.arange(Sqq.min(), Sqq.max(), 1)[:, np.newaxis]

regr = regr.fit(Sqq.reshape(-1, 1), Prq.reshape(-1, 1))
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(Prq.reshape(-1, 1), regr.predict(Sqq.reshape(-1, 1)))


# In[49]:


regr1 = LinearRegression()

X_quad1 = quadratic.fit_transform(nRoomq.reshape(-1, 1))
X_cubic1 = cubic.fit_transform(nRoomq.reshape(-1, 1))


# In[50]:


X_fit1 = np.arange(nRoomq.min(), nRoomq.max(), 1)[:, np.newaxis]

regr1 = regr1.fit(nRoomq.reshape(-1, 1), Prq.reshape(-1, 1))
y_lin_fit1 = regr1.predict(X_fit1)
linear_r21 = r2_score(Prq.reshape(-1, 1), regr1.predict(nRoomq.reshape(-1, 1)))


# In[51]:


regr2 = LinearRegression()

X_quad2 = quadratic.fit_transform(PbSq.reshape(-1, 1))
X_cubic2 = cubic.fit_transform(PbSq.reshape(-1, 1))


# In[52]:


X_fit2 = np.arange(PbSq.min(), PbSq.max(), 1)[:, np.newaxis]

regr2 = regr2.fit(PbSq.reshape(-1, 1), Prq.reshape(-1, 1))
y_lin_fit2 = regr2.predict(X_fit2)
linear_r22 = r2_score(Prq.reshape(-1, 1), regr2.predict(PbSq.reshape(-1, 1)))


# In[53]:


regr3 = LinearRegression()

quadratic3 = PolynomialFeatures(degree=2)
cubic3 = PolynomialFeatures(degree=3)
X_quad3 = quadratic.fit_transform(StageHq.reshape(-1, 1))
X_cubic3 = cubic.fit_transform(StageHq.reshape(-1, 1))


# In[54]:


X_fit3 = np.arange(StageHq.min(), StageHq.max(), 1)[:, np.newaxis]

regr3 = regr3.fit(StageHq.reshape(-1, 1), Prq.reshape(-1, 1))
y_lin_fit3 = regr3.predict(X_fit3)
linear_r23 = r2_score(Prq.reshape(-1, 1), regr3.predict(StageHq.reshape(-1, 1)))


# In[55]:


regr4 = LinearRegression()

quadratic4 = PolynomialFeatures(degree=2)
cubic4 = PolynomialFeatures(degree=3)
X_quad4 = quadratic.fit_transform(StageFq.reshape(-1, 1))
X_cubic4 = cubic.fit_transform(StageFq.reshape(-1, 1))


# In[56]:


X_fit4 = np.arange(StageFq.min(), StageFq.max(), 1)[:, np.newaxis]

regr4 = regr4.fit(StageFq.reshape(-1, 1), Prq.reshape(-1, 1))
y_lin_fit4 = regr4.predict(X_fit4)
linear_r24 = r2_score(Prq.reshape(-1, 1), regr4.predict(StageFq.reshape(-1, 1)))


# In[57]:


plt.figure(figsize=(16,10), dpi= 80)

plt.subplot(2, 3, 1)
plt.scatter(Sq, Pr, label='training points', color='lightblue')
plt.plot(X_fit, y_lin_fit, 
         label='линейный (d=1), $R^2={:.2f}$'.format(linear_r2), 
         color='blue', 
         lw=2, 
         linestyle='-')

plt.title("Площадь-Цена")
plt.legend(loc='upper right')


plt.subplot(2, 3, 2)
plt.scatter(nRomm, Prr, label='training points', color='lightblue')
plt.plot(X_fit1, y_lin_fit1, 
         label='линейный (d=1), $R^2={:.2f}$'.format(linear_r2), 
         color='blue', 
         lw=2, 
         linestyle='-')

plt.title("кол-во комнат-Цена")
plt.legend(loc='upper right')


plt.subplot(2, 3, 4)
plt.scatter(PbS, Pr, label='training points', color='lightblue')
plt.plot(X_fit2, y_lin_fit2, 
         label='линейный (d=1), $R^2={:.2f}$'.format(linear_r2), 
         color='blue', 
         lw=2, 
         linestyle='-')

plt.title("Цена за квадрат-Цена")
plt.legend(loc='upper right')


plt.subplot(2, 3, 5)
plt.scatter(StageH, Pr, label='training points', color='lightblue')
plt.plot(X_fit3, y_lin_fit3, 
         label='линейный (d=1), $R^2={:.2f}$'.format(linear_r2), 
         color='blue', 
         lw=2, 
         linestyle='-')

plt.title("этажность-Цена")
plt.legend(loc='upper right')


plt.subplot(1, 3, 3)
plt.scatter(StageF, Pr, label='training points', color='lightblue')
plt.plot(X_fit4, y_lin_fit4, 
         label='линейный (d=1), $R^2={:.2f}$'.format(linear_r2), 
         color='blue', 
         lw=2, 
         linestyle='-')

plt.title("этаж-Цена")
plt.legend(loc='upper right')

plt.show()


# In[ ]:




