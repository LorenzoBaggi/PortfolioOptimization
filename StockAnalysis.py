# -*- coding: utf-8 -*-
"""
Stock Market Analysis with Pandas Python
"""

import pandas_datareader.data as web
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

# use this or %matplotlib qt %matplotlub inline

# Just to show something about a company

# start = datetime.datetime(2020,1,1)
# end = datetime.datetime(2020,4,17)
# google = web.DataReader("GOOGL",'yahoo',start,end)
# plt.figure()
# google['Open'].plot(label = 'GOOGL Open Price', figsize=(15,7))
# google['Close'].plot(label = 'GOOGL Close Price')
# google['High'].plot(label = 'GOOGL High Price')
# google['Low'].plot(label = 'GOOGL Low Price')
# plt.legend()
# plt.title('Google Stock Prices')
# plt.ylabel('Stock Price')
# plt.show()
# plt.figure()
# google['Volume'].plot(figsize=(17,5))
# plt.title('Volume Traded by Google')

start = datetime.datetime(2010,1,1)
end = datetime.datetime(2022,5,30)
tlt  = web.DataReader("TLT",'yahoo',start,end)
spy  = web.DataReader("spy",'yahoo',start,end)
gld  = web.DataReader("gld",'yahoo',start,end)

#To save data in your folder, use: tesla.to_csv('Teslta.csv')

plt.figure()
tlt['Open'].plot(label = 'TLT Open Price', figsize=(15,7))
gld['Open'].plot(label = 'gld Open Price')
spy['Open'].plot(label = 'spy Open Price')
plt.ylabel('Stock Price')
plt.legend()

plt.figure()
tlt['Volume'].plot(label='TLT Volume', figsize=(17,5))
gld['Volume'].plot(label='gld volume', figsize=(17,5))
spy['Volume'].plot(label='spy volume', figsize=(17,5))
plt.ylabel('Volume traded')
plt.legend()

#Have a look at the total marketcap of the companies,
#we will use a kind of proxy, the produc of open price * volume traded

tlt['Total Traded']=tlt['Open']*tlt['Volume']
gld['Total Traded']=gld['Open']*gld['Volume']
spy['Total Traded']=spy['Open']*spy['Volume']

plt.figure()
tlt['Total Traded'].plot(label = 'TLT MarketCap', figsize=(15,7))
gld['Total Traded'].plot(label = 'gld MarketCap', figsize=(15,7))
spy['Total Traded'].plot(label = 'spy MarketCap', figsize=(15,7))
plt.ylabel('Total traded')
plt.legend()

maxspikeposition = tlt['Total Traded'].argmax() 
#2638
maxspikeday = tlt.iloc[[tlt['Total Traded'].argmax()]]
#2020-12-18, what happend on that very day 

#Moving Average
plt.figure()
gld['Open'].plot(label = 'gld Open Price', figsize=(15,7))
plt.ylabel('gld Price')
plt.legend()
#You can see a lot of noise, therefore moving averages are quite helpful
gld['MA50'] = gld['Open'].rolling(50).mean()
gld['MA9'] = gld['Open'].rolling(9).mean()
gld['MA200'] = gld['Open'].rolling(200).mean()
plt.figure()

gld['Open'].plot(label = 'gld Open Price', figsize=(15,7))
gld['MA50'].plot(label = 'gld MA50 Price')
gld['MA9'].plot(label = 'gld MA9 Price')
gld['MA200'].plot(label = 'gld MA200 Price')
plt.ylabel('gld Price')
plt.legend()


from pandas.plotting import scatter_matrix

car_comp = pd.concat([tlt['Open'], gld['Open'], spy['Open']], axis = 1)
car_comp.columns = ['TLT', 'gld', 'spy']
scatter_matrix(car_comp, figsize=(15,7), hist_kwds={'bins' : 30})
#you can see if you have a linear correlation between different variables


from mpl_finance import candlestick_ohlc
from matplotlib.dates import DateFormatter, date2num, WeekdayLocator, DayLocator, MONDAY

spy_reset = spy.loc['2012-01':'2012-01'].reset_index()

spy_reset['date_ax'] = spy_reset['Date'].apply(lambda date: date2num(date))
spy_values = [tuple(vals) for vals in spy_reset[['date_ax', 'Open', 'High', 'Low', 'Close']].values]

mondays = WeekdayLocator(MONDAY)
alldays = DayLocator()
WeekFormatter = DateFormatter('%b %d')
dayFormatter = DateFormatter ('%d')

fig, ax = plt.subplots()
candlestick_ohlc(ax, spy_values, width = 0.6, colorup='b', colordown='r')

#Daily Percentage Change
tlt['returns'] = (tlt['Close']/tlt['Close'].shift(1)) - 1 
gld['returns'] = (gld['Close']/gld['Close'].shift(1)) - 1 
spy['returns'] = (spy['Close']/spy['Close'].shift(1)) - 1 

plt.figure()
#The fatter it is, the more volatile it is 
spy['returns'].hist(bins=50, label = 'spy', alpha = 0.5, figsize = (15,7))
gld['returns'].hist(bins=50, label = 'gld', alpha = 0.5)
tlt['returns'].hist(bins=50, label = 'TLT', alpha = 0.5)
plt.legend()

#The Tesla Curve is wider than GM and Ford, TSLA is more volatile

#The Kernel density estimate plot
plt.figure()
tlt['returns'].plot(kind='kde', label = 'TLT', figsize = (15,8))
spy['returns'].plot(kind='kde', label = 'spy')
gld['returns'].plot(kind='kde', label = 'gld')
plt.legend()

#Visualize some Box Plots
#really informative

box_df = pd.concat([tlt['returns'],gld['returns'],spy['returns']], axis = 1)
box_df.columns=('TLT Returns', 'gld Returns', 'spy Returns')
box_df.plot(kind='box')


scatter_matrix(box_df, figsize=(10,10),hist_kwds={'bins':50}, alpha = 0.5)

#Cumulative Return
tlt['Cumulative Return'] = (1+ tlt['returns']).cumprod()
gld['Cumulative Return'] = (1+ gld['returns']).cumprod()
spy['Cumulative Return'] = (1+ spy['returns']).cumprod()

#Allocate your Portfolio

ptf = tlt

ptf['Cumulative Return'] = 0.6*((1+spy['returns']).cumprod()) + 0.25*((1+tlt['returns']).cumprod()) + 0.15*((1 + gld['returns']).cumprod())

plt.figure()
tlt['Cumulative Return'].plot(label='TLT', figsize = (15,7))
gld['Cumulative Return'].plot(label='gld')
spy['Cumulative Return'].plot(label='spy')
ptf['Cumulative Return'].plot(label='portfolio')
plt.title('Cumulative Return vs')
plt.legend()