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

end = datetime.date.today()

begin=end-pd.DateOffset(365*10)

st=begin.strftime('%Y-%m-%d')

ed=end.strftime('%Y-%m-%d')


data = pdr.get_data_yahoo("^NSEI",st,ed)

data.dropna(inplace=True)

lprice=np.log(data['Close'])

ret = (lprice/lprice.shift(1)).dropna()

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

'''Simulating white noise series'''

np.random.seed(1)

d=np.random.normal(size=1000)

tsplot(d,lags=30)

'''Simulating a random walk'''

np.random.seed(1)

n_samples=1000

x=w=np.random.normal(size=n_samples)

for t in range(n_samples):
    x[t]=x[t-1]+w[t]

_=tsplot(x,lags=30)

_=tsplot(np.diff(x),lags=30)

'''Simulating AR(1) with alpha 0.6'''

np.random.seed(1)

n_samples=1000

a=0.6

x=w=np.random.normal(size=n_samples)

for t in range(n_samples):
    x[t]=a*x[t-1]+w[t]
    
_=tsplot(x,lags=30)  

mdl= smt.AR(x).fit(maxlag=30,ic='aic',trend='nc')
'''ic = information criteria'''
mdl.params[0]

print(mdl.summary())

order=smt.AR(x).select_order(maxlag=30,ic='aic',trend='nc')



from statsmodels.tsa.stattools import adfuller

dftest=adfuller(lprice,autolag='AIC')

dftest2=adfuller(ret,autolag='AIC')


best_aic=np.inf

best_order=None

best_mdl=None


rng=range(5) # [0,1,2,3,4]

for i in rng:
    for j in rng:
        try:
            tmp_mdl=smt.ARMA(ret,order=(i,j)).fit(method='mle',trend='nc')
            tmp_aic=tmp_mdl.aic
            if tmp_aic<best_aic:
                best_aic=tmp_aic
                best_order=(i,j)
                best_mdl=tmp_mdl
        except:
            continue
        
kk=best_mdl.resid


f,err95,ci95=best_mdl.forecast(steps=20)

f=pd.DataFrame(f)
