import yfinance as yf
import pandas as pd
import pandas_datareader.data as reader
import datetime as dt
import statsmodels.api as sm
import matplotlib.pyplot as plt
import getFamaFrenchFactors as gff
import seaborn as sns
import numpy as np



## Import stock prices 

end = dt.datetime.now()
start = dt.date(end.year - 3, end.month, end.day)

tickers = ['BATT.MI', 'HTWO.MI', 'CNYA', 'GLD', 'SWDA.MI']
benchmark = ['SPY', 'TLT']

df_raw = yf.download(tickers, start, end)['Adj Close']
df = df_raw.ffill().bfill()

df_bm_raw = yf.download(benchmark, start, end)['Adj Close']
df_bm = df_bm_raw.ffill().bfill()


# resampling 

df = df.resample('1M').last()
df_bm = df_bm.resample('1M').last()

# calculate the returns

df_returns = df.pct_change()
df_returns.iloc[0,] = 0

df_returns_bm = df_bm.pct_change()
df_returns_bm.iloc[0,] = 0


## Calculating the portfolio returns

weigths = np.array([0.05, 0.05, 0.05, 0.05, 0.8])
df_returns['Portfolio_monthly_returns'] = df_returns.dot(weigths)

weigths_bm = np.array([0.6, 0.4])
df_returns_bm['Benchmark_monthly_returns'] = df_returns_bm.dot(weigths_bm)

cum_returns = (1+df_returns).cumprod()
cum_returns_bm = (1+df_returns_bm).cumprod()

# plotting

plt.plot(cum_returns['Portfolio_monthly_returns'], label = 'Portfolio_monthly_returns' )
plt.plot(cum_returns_bm['Benchmark_monthly_returns'], label = 'Benchmark_monthly_returns')
plt.legend()
plt.xticks(rotation=30)
plt.show()


# market excess returns

ff3_monthly = pd.DataFrame(gff.famaFrench3Factor(frequency='m'))
ff3_monthly.rename(columns={'date_ff_factors' : 'Date'}, inplace=True)
ff3_monthly.set_index('Date', inplace=True)

# merging dataframes

data = ff3_monthly.merge(df_returns, on = 'Date').drop(columns = ["BATT.MI", "HTWO.MI", "CNYA", "GLD", "SWDA.MI"])

ptf_excess_return = data['Portfolio_monthly_returns']- data['RF']
data['Excess_Return'] = ptf_excess_return

sns.regplot(x='Mkt-RF', y ='Excess_Return', data=data)

# regression

x = data['Mkt-RF']
y = data['Excess_Return']

X1 = sm.add_constant(x)
model = sm.OLS(y,X1)

results = model.fit()
print(results.summary())

# my portfolio sadly does not have alpha 
