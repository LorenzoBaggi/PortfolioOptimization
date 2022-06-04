# -*- coding: utf-8 -*-
"""
@author: LorenzoBaggi
The script is created relying on:
https://pypi.org/project/pyportfolioopt/
"""

"Insert the tickers that you want, insert the time-spane and you'll find"
"value at risk, efficient frontier, correlation matrix, maxsharpe weights"
"and performances, min volatility weights and performances, target return"
"weigths and performances."
"Below you'll find a slightly modified version of the Buy and Hold Golden"
"Butterfly portfolio"."

import pandas as pd
import yfinance as yf
import numpy as np
from pandas_datareader import data as pdr
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

# Performance of your Portfolio
print("Elaborating ... ")
TICKERS =['SHY', 'TLT', 'VTI', 'IWN', 'GLD', 'BTC-USD']

def getData(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockData= stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov() 
    return meanReturns, covMatrix

def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*252
    std = np.sqrt(
            np.dot(weights.T,np.dot(covMatrix, weights))
           )*np.sqrt(252)
    return returns, std
    
stocks = [stock for stock in TICKERS]
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

# Choouse the investment period of your portfolio
endDate = dt.datetime(2022,5,27)
startDate = dt.datetime(1997,1,1)

meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)
returns, std = portfolioPerformance(weights, meanReturns, covMatrix)

SR = (returns)/std
print("The SR of the provided portfolio, with Risk Free Rate = 1.02% is:", SR)

# Here comes the optimization process 
INVESTMENT = 10000 

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

stocks_prices = pd.DataFrame({'A' : []})
stocks_info = pd.DataFrame({'A' : []})

for i,ticker in enumerate(TICKERS):
  print(i,ticker)
  yticker = yf.Ticker(ticker)
  
  # Get max history of prices
  #historyPrices = yticker.history(period='max')
  historyPrices = yticker.history(start="1997-01-01", end="2022-05-27")
  # Generate features for historical prices, and what we want to predict

  historyPrices['Ticker'] = ticker
  historyPrices['Year']= historyPrices.index.year
  historyPrices['Month'] = historyPrices.index.month
  historyPrices['Weekday'] = historyPrices.index.weekday
  historyPrices['Date'] = historyPrices.index.date
  
  # historical returns
  for i in [1,3,7,30,90,365]:
    historyPrices['growth_'+str(i)+'d'] = historyPrices['Close'] / historyPrices['Close'].shift(i)

  # future growth 3 days  
  historyPrices['future_growth_3d'] = historyPrices['Close'].shift(-3) / historyPrices['Close']

  # 30d rolling volatility : https://ycharts.com/glossary/terms/rolling_vol_30
  historyPrices['volatility'] =   historyPrices['Close'].rolling(30).std() * np.sqrt(252)

  if stocks_prices.empty:
    stocks_prices = historyPrices
  else: 
    stocks_prices = pd.concat([stocks_prices,historyPrices], ignore_index=True)
    
filter_last_date = stocks_prices.Date==stocks_prices.Date.max()
print(stocks_prices.Date.max())

df = stocks_prices.pivot('Date','Ticker','Close').reset_index()
# print(df.tail(5))

# Correlation matrix
df.corr()
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, annot=True, cmap='RdYlGn')
   
from pypfopt import risk_models
from pypfopt import plotting

from pypfopt import expected_returns
from pypfopt import EfficientFrontier

# json: for pretty print of a dictionary: https://stackoverflow.com/questions/44689546/how-to-print-out-a-dictionary-nicely-in-python/44689627
import json

mu = expected_returns.capm_return(df.set_index('Date'))
# Other options for the returns values: expected_returns.ema_historical_return(df_pivot.set_index('Date'))
# Other options for the returns values: expected_returns.mean_historical_return(df_pivot.set_index('Date'))
print(f'Expected returns for each stock: {mu} \n')

S = risk_models.CovarianceShrinkage(df.set_index('Date')).ledoit_wolf()

# Weights between 0 and 1 - we don't allow shorting
ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.5))
ef.min_volatility()
weights_min_volatility = ef.clean_weights()

print(f'Portfolio weights for min volatility optimisation (lowest level of risk): {json.dumps(weights_min_volatility, indent=4, sort_keys=True)} \n')
print(f'Portfolio performance: {ef.portfolio_performance(verbose=True, risk_free_rate=0.00)} \n')
# Risk-free rate : 10Y TBonds rate on 21-Jul-2021 https://www.cnbc.com/quotes/US10Y
ef.portfolio_performance(verbose=False, risk_free_rate=0.00); 
plt.figure()
pd.Series(weights_min_volatility).plot.barh(title = 'Optimal Portfolio Weights (min volatility) by PyPortfolioOpt');

ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.5))
ef.max_sharpe(risk_free_rate=0.00)
weights_max_sharpe = ef.clean_weights()

print(f'Portfolio weights for max Sharpe): {json.dumps(weights_max_sharpe, indent=4, sort_keys=True)} \n')
print(f'Portfolio performance: {ef.portfolio_performance(verbose=True, risk_free_rate=0.00)} \n')
weight_arr = ef.weights
ef.portfolio_performance(verbose=False, risk_free_rate=0.00); 
plt.figure()
pd.Series(weights_max_sharpe).plot.barh(title = 'Optimal Portfolio Weights (max Sharpe) by PyPortfolioOpt');

ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.5))
ef.efficient_return(0.2, market_neutral=False)
weights_max_return = ef.clean_weights()

print(f'Portfolio weights for max Return): {json.dumps(weights_max_return, indent=4, sort_keys=True)} \n')
print(f'Portfolio performance: {ef.portfolio_performance(verbose=True, risk_free_rate=0.00)} \n')
ef.portfolio_performance(verbose=False, risk_free_rate=0.00); 
plt.figure()
pd.Series(weights_max_return).plot.barh(title = 'Optimal Portfolio Weights (max Return) by PyPortfolioOpt');


returns = expected_returns.returns_from_prices(df.set_index('Date')).dropna()
returns.head()

plt.figure()
portfolio_rets = (returns * weight_arr).sum(axis=1) 
portfolio_rets.hist(bins=50)

plt.show()

# VaR
var = portfolio_rets.quantile(0.05)
cvar = portfolio_rets[portfolio_rets <= var].mean()
print("VaR: {:.2f}%".format(100*var))
print("CVaR: {:.2f}%".format(100*cvar))


from pypfopt import CLA, plotting

cla = CLA(mu, S, weight_bounds=(0, 0.5))
cla.max_sharpe()
# cla.portfolio_performance(verbose=True, risk_free_rate=0.00);

plt.figure()
ax = plotting.plot_efficient_frontier(cla, showfig=False)

n_samples = 1000
w = np.random.dirichlet(np.ones(len(mu)), n_samples)
rets = w.dot(mu)
stds = np.sqrt((w.T * (S @ w.T)).sum(axis=0))
sharpes = rets / stds

print("Sample portfolio returns:", rets)
print("Sample portfolio volatilities:", stds)

# Plot efficient frontier with Monte Carlo sim
ef = EfficientFrontier(mu, S)

fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)



ef = EfficientFrontier(mu, S)
ef.max_sharpe()
weight_arr = ef.weights
ret_tangent, std_tangent, _ = ef.portfolio_performance()
# Plot random portfolios
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")



# Format
ax.set_title("PyPortfolioOpt: Efficient Frontier with random portfolios")
ax.legend()

plt.tight_layout()
plt.show()


# Data to plot
plt.figure()
ax.set_title("Pie Chart of Optimal Portfolio Weights")
plt.title('Pie Chart for Max Sharpe')
plt.pie(weights_max_sharpe .values(), labels=weights_max_sharpe.keys(), autopct='%1.1f%%', shadow=True, startangle=90, normalize=False)
