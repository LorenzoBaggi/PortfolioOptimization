#importing 
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

start_array = ['2015-12-30', '2016-04-30', '2016-06-30', '2016-08-30', '2016-12-30', '2017-04-30', '2017-06-30',\
     '2017-08-30', '2017-12-30', '2018-04-30', '2018-06-30', '2018-08-30',\
     '2018-12-30', '2019-04-30', '2019-06-30', '2019-08-30', '2019-12-30',\
        '2020-04-30','2020-06-30', '2020-08-30', '2020-12-30']
df_total = pd.DataFrame()
LS_ret = []
DCA_ret = []

#dowloading data
for i in range(0,len(start_array)):
    df = yf.download('DOGE-USD', start = start_array[i])
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

    LS_ret_i = (wealth_LS[-1]/wealth_LS[0] - 1 ) * 100
    LS_ret.append(LS_ret_i)
    DCA_ret_i = (wealth_DCA[-1]/wealth_LS[0] - 1 ) * 100
    DCA_ret.append(DCA_ret_i)

    df_recap = pd.concat([wealth_DCA, wealth_LS], axis = 1)
    df_total = pd.concat([df_total, df_recap], axis = 1)


#df_total.to_excel('appended.xlsx')
print(LS_ret)
print(DCA_ret)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

n, bins, patches = ax1.hist(LS_ret)
ax1.set_xlabel('LS ret')
ax1.set_ylabel('Frequency')

n, bins, patches = ax2.hist(DCA_ret)
ax2.set_xlabel('DCA ret')
ax2.set_ylabel('Frequency')
plt.show()


