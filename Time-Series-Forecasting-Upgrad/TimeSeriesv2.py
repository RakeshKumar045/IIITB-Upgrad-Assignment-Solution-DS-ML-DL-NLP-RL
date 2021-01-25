#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa as ts
from matplotlib import pyplot
from scipy.stats import mstats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import VARMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.vecm import select_coint_rank

# In[2]:


########## reading the data and preprocessing for current problem#################
inputData = pd.read_csv('inputDataProcessed.csv', encoding="ISO-8859-1")

inputData.head()

# In[25]:


n = len(pd.unique(inputData['Store']))
print(n)

i = 9

# In[26]:


# plot sales data
storeDataAll = inputData.loc[inputData["Store"] == i]
storeDataAll = storeDataAll.reset_index(drop=True)

# Create figure and plot space

plt.plot(storeDataAll.iloc[0:250, 1], storeDataAll.iloc[0:250, 2])

# In[6]:


plt.scatter(storeDataAll['Sales'], storeDataAll['Customers'])

# In[27]:


# cap sales Data
transformed_test_data1 = pd.Series(mstats.winsorize(storeDataAll['Sales'], limits=[0.05, 0.05]))
transformed_test_data1.plot()

# In[28]:


transformed_test_data2 = pd.Series(mstats.winsorize(storeDataAll['Customers'], limits=[0.05, 0.05]))
transformed_test_data2.plot()

# In[29]:


# get transformed values
storeDataAll.iloc[:, 2] = transformed_test_data1.values
storeDataAll.iloc[:, 3] = transformed_test_data2.values

# In[30]:


################# get seasonality and trend ##########################
##### change codde to include search for best sesonality using freq#############

result = seasonal_decompose((storeDataAll.iloc[:, 2]), model='additive', freq=365, extrapolate_trend='freq')
result.plot()

# plot_acf(storeDataAll.iloc[:,2])


# In[31]:


print(result.resid.mean())

storeDataAll["SalesSeasonality"] = result.seasonal
storeDataAll["SalesTrend"] = result.trend

# In[32]:


################# get seasonality and trend ##########################
##### change codde to include search for best sesonality using freq#############
result = seasonal_decompose((storeDataAll.iloc[:, 3]), model='additive', freq=365, extrapolate_trend='freq')
result.plot()
print(result.resid.mean())
# plot_acf(storeDataAll.iloc[:,3])
storeDataAll["CustSeasonality"] = result.seasonal
storeDataAll["CustTrend"] = result.trend

# In[33]:


###### split data into train and test #########################################
storeOneData, TestData = storeDataAll[:-100], storeDataAll[-100:]
storeOneData.shape
TestData.shape

# In[34]:


#################### causality test####################################
## null hypothesis is: x does not granger cause y #####################
## if value of p is less than 0.05 then granger causality exists ######

CausalitySales = (ts.stattools.grangercausalitytests(storeOneData[['Sales', 'Customers']].dropna(), 1))
# print(CausalitySales)
CausalityCust = (ts.stattools.grangercausalitytests(storeOneData[['Customers', 'Sales']].dropna(), 1))
# print(CausalityCust)


# In[35]:


####################### stationarity ##################################
## null hypothesis is there is nonstationarity ########################
## if p<0.05 then series is staionary no differencing reqd ############

station = adfuller(storeOneData.iloc[:, 2], autolag='AIC')
print('ADF Statistic: %f' % station[0])
print('p-value: %f' % station[1])

if station[4]['5%'] < station[0]:
    stationDIF = adfuller(storeOneData.iloc[:, 3].diff().dropna(), autolag='AIC')
    print('ADF Statistic DIFF: %f' % stationDIF[0])
    print('p-value DIFF: %f' % stationDIF[1])

print(station)

# In[36]:


####################### stationarity ##################################
## null hypothesis is there is nonstationarity ########################
## if p<0.05 then series is staionary no differencing reqd ############

station = adfuller(storeOneData.iloc[:, 3], autolag='AIC')
print('ADF Statistic: %f' % station[0])
print('p-value: %f' % station[1])

if station[4]['5%'] < station[0]:
    stationDIF = adfuller(storeOneData.iloc[:, 3].diff().dropna(), autolag='AIC')
    print('ADF Statistic DIFF: %f' % stationDIF[0])
    print('p-value DIFF: %f' % stationDIF[1])

print(station)

# In[37]:


# define endogenous and exogenous variables

endog = storeOneData[['Sales', 'Customers']].astype('float32')

exog = storeOneData[['Promo', 'SchoolHoliday',
                     'DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3',
                     'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_0', 'SalesSeasonality',
                     'SalesTrend', 'CustSeasonality', 'CustTrend']]
exog = exog.astype('float32')
endog = endog.astype('float32')

# In[79]:


# get the differenced series if required

endogdif1 = endog.diff().dropna()
exogdif1 = exog.diff().dropna()

endogdif11 = endog.iloc[:, 0].diff().dropna()
endogdif1.iloc[:, 0] = endogdif11.values

endogdif12 = endog.iloc[:, 1].diff().dropna()
endogdif1.iloc[:, 1] = endogdif12.values

# TestDatadif1=TestData.diff().dropna()


# In[39]:


#################### cointegration Analysis if required###################################
## null hypothesis is that there is no cointegration ##########################
## to be tested if ADF test says non-stationarity #############################

coint = coint_johansen(endogdif1, 0, 1)
coint.trace_stat
coint.max_eig_stat
traces = coint.lr1
maxeig = coint.lr2
cvts = coint.cvt  ## 0: 90%  1:95% 2: 99%
cvms = coint.cvm  ## 0: 90%  1:95% 2: 99%

N, l = endogdif1.shape

for i in range(l):
    if traces[i] > cvts[i, 1]:
        r = i + 1
print(r)

rank = select_coint_rank(endogdif1, 0, 1)
print(rank.rank)

# mod = VECM(endogdif1, exog=exogdif1) #endogdif1, exogdif1
# res = mod.fit() #maxiter=500, disp=False
# res.hessian()
# print(res.summary())


# In[71]:


# specify VAR model at levels
mod = VAR(endogdif1, exog=exogdif1)  # , order=(2,0,0)

aa = mod.select_order()
aa.summary
print(aa.aic)
res = mod.fit(maxlags=aa.aic, ic='aic')
lag_order = res.k_ar
res.summary()

# In[ ]:


# In[60]:


# specify VARMAX model

mod = VARMAX(endogdif1, exog=exogdif1, order=(18, 0), trend='n')  # endogdif1, exogdif1
res = mod.fit(maxiter=100, disp=False)  # maxiter=500, disp=False
# res.hessian()
print(res.summary())
print(res.params)

# In[96]:


# get impulse response functions
irf = res.impulse_responses(steps=100, orthogonalized=False)
irf.plot()
# res.plot_diagnostics(figsize=(16, 8))


# In[95]:


# forecast and get the accuracy of the forecast for differenced series

forecast_input = endogdif1.values[-lag_order:]

forcast = res.forecast(y=forecast_input, steps=678, exog_future=exogdif1)
forcast = pd.DataFrame(forcast)
forcast.head()
forcast.iloc[:, 0].plot()

actual = endogdif1[["Sales"]]

actual.columns = ['ActualSales']
actual = actual.reset_index(drop=True)
predicted = forcast.iloc[:, 0]
predicted = predicted.reset_index(drop=True)
pred = pd.merge(actual, predicted, right_index=True, left_index=True)
pred = pred[pred.ActualSales != 0]

pred.columns = ['ActualSales', 'Sales']
pred.head()

# plot
pyplot.plot(actual)
pyplot.plot(predicted, color='red')
pyplot.show()

MPE = np.mean((pred.ActualSales - pred.Sales) / (pred.ActualSales))
print(MPE)

MAPE = np.mean(abs(pred.ActualSales - pred.Sales) / (pred.ActualSales))
print(MAPE)

# In[63]:


# forecast and get the accuracy of the forecast
forcast = res.forecast(steps=len(endog), exog=exog)
# forcast.iloc[:,0].plot()

actual = endog[["Sales"]]

actual.columns = ['ActualSales']
actual = actual.reset_index(drop=True)
predicted = forcast[['Sales']]
predicted = predicted.reset_index(drop=True)
pred = pd.merge(actual, predicted, right_index=True, left_index=True)
pred = pred[pred.ActualSales != 0]

# plot
pyplot.plot(actual)
pyplot.plot(predicted, color='red')
pyplot.show()

MPE = np.mean((pred.ActualSales - pred.Sales) / (pred.ActualSales))
print(MPE)

MAPE = np.mean(abs(pred.ActualSales - pred.Sales) / (pred.ActualSales))
print(MAPE)

# In[84]:


##################### forecast on test sample ##############################

endogTest = TestData[['Sales', 'Customers']].astype('float32')

exogTest = TestData[['Promo', 'SchoolHoliday',
                     'DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3',
                     'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_0', 'SalesSeasonality',
                     'SalesTrend', 'CustSeasonality', 'CustTrend']]
exogTest = exogTest.astype('float32')
endogTest = endogTest.astype('float32')

forcast = res.forecast(steps=len(endogTest), exog=exogTest)
# forcast.iloc[:,0].plot()

actual = endogTest[["Sales"]]

actual.columns = ['ActualSales']
actual = actual.reset_index(drop=True)
predicted = forcast[['Sales']]
predicted = predicted.reset_index(drop=True)
pred = pd.merge(actual, predicted, right_index=True, left_index=True)
pred = pred[pred.ActualSales != 0]

# plot
pyplot.plot(actual)
pyplot.plot(predicted, color='red')
pyplot.show()

MPE = np.mean((pred.ActualSales - pred.Sales) / (pred.ActualSales))
print(MPE)

MAPE = np.mean(abs(pred.ActualSales - pred.Sales) / (pred.ActualSales))
print(MAPE)

# In[ ]:
