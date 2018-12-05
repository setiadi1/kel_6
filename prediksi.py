import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
import pandas as pd
import numpy as np
matplotlib.use('Agg')

from pylab import rcParams

rcParams['xtick.major.pad']='4'
rcParams['ytick.major.pad']='4'
rcParams['lines.linewidth'] = 1
rcParams['savefig.facecolor'] = "1"
rcParams['axes.facecolor']= "1"
rcParams["axes.edgecolor"] = "1"

def init(s):
#     d = range(0,2)
#     p = q = range(0,5)
#     P = Q = D = range(0,2)
        
    # pdq = list(itertools.product(p,d,q))
    # seasonal_pdq_x = list(itertools.product(P,D,Q))
    # seasonal_pdq = [(x[0], x[1], x[2], s) for x in seasonal_pdq_x]
    
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], s) for x in pdq]    
    
    return pdq, seasonal_pdq

def calc(ts, s, n):
    pdq, seasonal_pdq = init(s)
    results_table = bestAIC(ts,pdq,seasonal_pdq)
    result, result_summary = Fit(ts,results_table)

    # plot_prediksi, prediksi = predict(ts, result)
    # MSE, RMSE = error(prediksi)
    # forecast = Forecast(ts, result, n)
    # return result_summary, plot_prediksi, MSE, RMSE, forecast

    return result_summary, result

def bestAIC(ts, pdq, seasonal_pdq):
    results_aic = []
    best_aic = float("inf")
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(ts, order=param, seasonal_order=param_seasonal, enforce_invertibility=False, enforce_stationarity=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
            if results.aic < best_aic:
                best_aic = results.aic
            results_aic.append([param,param_seasonal, results.aic]) 
    result_table = pd.DataFrame(results_aic)
    result_table.columns = ['parameters','seasonal_param', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    return result_table
    
def Fit(ts, results_table):
    p, d, q = results_table.parameters[0]
    P, D, Q, s = results_table.seasonal_param[0]

    mod = sm.tsa.statespace.SARIMAX(ts, order=(p, d, q), seasonal_order=(P, D, Q, s), enforce_invertibility=False, enforce_stationarity=False)
    results = mod.fit()
    return results, results.summary()

def predict(ts, result):
    pred = result.get_prediction(start = 0)
    pred_ci = pred.conf_int()
    fig, ax = plt.subplots(1)
    ax = ts.plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    # plt.legend()
    # return plt.show, pred
    return pred

def error(ts, pred):
    y_forecasted = pred.predicted_mean
    y_truth = ts
    mse = ((y_forecasted - y_truth) ** 2).mean()
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_truth - y_forecasted) / y_truth)) *100
    return round(mse, 2), round(rmse, 2), round(mape,2)

def Forecast(ts, result, n):
    pred_uc = result.get_forecast(steps=n)
    pred_ci = pred_uc.conf_int()
    fig, ax = plt.subplots(1)
    ax = ts.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    plt.legend()
    return plt.show()