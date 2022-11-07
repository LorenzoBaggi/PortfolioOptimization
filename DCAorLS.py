#importing 
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

start_array = ['2012-12-31','2014-12-31','2015-12-31', '2016-12-31', '2017-12-31', '2018-12-31', '2020-12-31', '2021-12-31']
df_total = pd.DataFrame()

#dowloading data
for i in range(0,len(start_array)):
    df = yf.download('BTC-USD', start = start_array[i])
    # identifying the buying dates, assuming buying 1 per month
    # no more than 10k invested
    total_investment = 10000
    share = 500
    n_max_inv = total_investment / share

    buydates_old = pd.date_range(df.index[0], df.index[-1], freq='1M')
    if len(buydates_old) > n_max_inv:
        buydates = buydates_old[0:20]
    else:
        buydates = buydates_old

    # assuming to buy at the close price
    buyprices = df[df.index.isin(buydates)].Close

    # DCA sum
    share_bought_DCA = share/buyprices
    tot_share_bought_DCA = share_bought_DCA.cumsum()

    # amount bough w/ Lump Sum
    share_bought_LS = share * len(buyprices) / buyprices[0]

    # evaluating wealth
    tot_share_bought_DCA.name = 'share_amt_DCA'
    df_tog = pd.concat([tot_share_bought_DCA, df], axis=1).ffill()
    df_tog['share_amt_LS'] = share_bought_LS

    wealth_DCA = (df_tog.share_amt_DCA * df_tog.Close)/1000
    wealth_LS = (df_tog.share_amt_LS * df_tog.Close)/1000

    df_recap = pd.concat([wealth_DCA, wealth_LS], axis = 1)
    df_total = pd.concat([df_total, df_recap], axis = 1)

df_total.to_excel('appended.xlsx')
