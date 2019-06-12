# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 07:07:46 2017

@author:Ouito
"""

windowLength = 252

foreLength = len(ret) - windowLength
signal = 0*ret[-foreLength:]



for d in range(foreLength):
    
    # create a rolling window by selecting 
    # values between d+1 and d+T of Nifty-50 returns
    
    TS = ret[(1+d):(windowLength+d)] 
    
    # Find the best ARIMA fit 
    # set d = 0 since we've already taken log return of the series
    res_tup = best_arima(TS)
    order = res_tup[1]
    model = res_tup[2]
    
    #now that we have our ARIMA fit, we feed this to GARCH model
    p_ = order[0]
    o_ = order[1]
    q_ = order[2]
    
    am = arch_model(model.resid, p=p_, o=o_, q=q_, dist='StudentsT')
    res = am.fit(update_freq=5, disp='off')
    
    # Generate a forecast of next day return using our fitted model
    out = res.forecast(horizon=1, start=None, align='origin')
    
    #Set trading signal equal to the sign of forecasted return
    # Buy if we expect positive returns, sell if negative
      
    signal.iloc[d] = np.sign(out.mean['h.1'].iloc[-1])
    
signal=pd.Data
