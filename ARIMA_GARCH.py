import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import warnings
warnings.filterwarnings('ignore')


st="2015-01-01"

ed="2017-11-30"


data = pdr.get_data_yahoo("^NSEI",st,ed)

data.dropna(inplace=True)

#lprice=np.log(data['Close'])

ret = np.log(data['Close']/data['Close'].shift(1)).dropna()

def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return


from statsmodels.tsa.stattools import adfuller

dftest2 = adfuller(ret, autolag='AIC')


def best_arima(dataframe):
    best_aic = np.inf 
    best_order = None
    best_mdl = None

    pq_rng = range(5) # [0,1,2,3,4]
    d_rng = range(2) # [0,1]
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(dataframe, order=(i,d,j)).fit(
                        method='mle', trend='nc'
                    )
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))                    
    return best_aic, best_order, best_mdl

model = best_arima(ret)

order=model[1]

p=order[0]

d=order[1]

q=order[2]

print(p,d,q)

#How to see the residuals of ARIMA model?


from arch import arch_model

np.random.seed(1)

a0 = 2
a1 = .5

y = w = np.random.normal(size=1000)
Y = np.empty_like(y)

for t in range(len(y)):
    y[t] = w[t] * np.sqrt((a0 + a1*y[t-1]**2))

# simulated ARCH(1) series, looks like white noise
tsplot(y, lags=30)


tsplot(y**2, lags=30)


# Simulating a GARCH(1, 1) process

np.random.seed(2)

a0 = 0.2
a1 = 0.5
b1 = 0.3

n = 10000
w = np.random.normal(size=n)
eps = np.zeros_like(w)
sigsq = np.zeros_like(w)

for i in range(1, n):
    sigsq[i] = a0 + a1*(eps[i-1]**2) + b1*sigsq[i-1]
    eps[i] = w[i] * np.sqrt(sigsq[i])

_ = tsplot(eps, lags=30)

_ = tsplot(eps**2, lags=30)


am = arch_model(eps)
res = am.fit(update_freq=5)
print(res.summary())


#Fit an ARCH model

import warnings
warnings.filterwarnings('ignore')
vol_model = arch_model(model[2].resid)

res = vol_model.fit(update_freq=5)

print(res.summary())

#Interpreting the coefficients

#Fit a GARCH model

am = arch_model(model[2].resid, p=p, o=d, q=q, dist='StudentsT')

res = am.fit(update_freq=5, disp='off')
print(res.summary())

tsplot(res.resid,30)
