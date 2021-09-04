import pandas as pd 
import math
import numpy as np
from datetime import datetime, timedelta

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override
from datetime import datetime
import numpy as np
import pandas_datareader as web
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def plotog(df):
    plt.figure(figsize=(12.2, 4.5))
    plt.title('Close Price History')
    plt.plot(df['Close'], color = 'blue', alpha = 0.35)
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Close Price', fontsize = 18)
    plt.show()

#dma
def buy_selldma(data):
  sigPricebuy = []
  sigPricesell = []
  flag = -1

  for i in range (len(data)):
    if data['SMA30'][i] > data['SMA100'][i]: 
      if flag !=1:
        sigPricebuy.append(data['df'][i])
        sigPricesell.append(np.nan)
        flag = 1 
      else: 
       sigPricebuy.append(np.nan)
       sigPricesell.append(np.nan)
    elif data['SMA30'][i] < data['SMA100'][i]:
      if flag != 0:
        sigPricebuy.append(np.nan)
        sigPricesell.append(data['df'][i])
        flag = 0
      else:
        sigPricesell.append(np.nan)
        sigPricebuy.append(np.nan)
    else: 
      sigPricebuy.append(np.nan)
      sigPricesell.append(np.nan)


  return (sigPricebuy, sigPricesell)

def dmaplots(data):
  plt.figure(figsize=(12.6, 4.6))
  plt.plot(data['df'], alpha = 0.35)
  plt.plot(data['SMA30'], label ='SMA30', alpha = 0.35 )
  plt.plot(data['SMA100'], label = 'SMA100', alpha = 0.35)
  plt.scatter(data.index, data['Buy'], label = 'Buy', marker= '^', color = 'green')
  plt.scatter(data.index, data['Sell'], label = 'Sell', marker= 'v', color = 'red')
  plt.title('Dual Moving Average')
  plt.xlabel('Date')
  plt.xticks(rotation=45)
  plt.ylabel('Price')
  plt.legend(loc = 'lower left')
  plt.show()



def dma(df):
  SMA30=pd.DataFrame()
  SMA30['Close'] = df['Close'].rolling(window=30).mean()
  #SMA30
  SMA100=pd.DataFrame()
  SMA100['Close'] = df['Close'].rolling(window=100).mean()
  #SMA100
  data = pd.DataFrame()
  data['df'] = df['Close']
  data['SMA30'] = SMA30['Close']
  data['SMA100'] = SMA100['Close']
  #data
  a = buy_selldma(data)
  data['Buy'] = a[0]
  data['Sell'] = a[1]
  dmaplots(data)
  
  print(data.tail(5)[['df', 'Buy', 'Sell']])
  

def buy_sellmacd(signal):
    Buy=[]
    Sell=[]
    flag= -1
    for i in range(0, len(signal)):
        if signal['MACD'][i] > signal['Signal Line'][i]: 
            Sell.append(np.nan)
            if flag != 1:
                Buy.append(signal['Close'][i])
                flag =1 
            else: 
                Buy.append(np.nan)
        elif signal['MACD'][i] < signal['Signal Line'][i]: 
            Buy.append(np.nan)
            if flag != 0:
                Sell.append(signal['Close'][i])
                flag = 0 
            else: 
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)

    return(Buy, Sell) 

def macdPlot(df):
  #visually show buy and sell signals
  plt.figure(figsize=(12.6, 4.6))
  plt.scatter(df.index, df['Buy'], label='Buy', marker='^', alpha =1)
  plt.scatter(df.index, df['Sell'], label='Sell', marker='v', alpha =1)
  plt.plot(df['Close'], label='Close Price', alpha =0.35)
  plt.title("MACD")
  plt.xticks(rotation=45)
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend(loc = 'lower left')
  plt.show()

def fnMacd(df):
  #calculations for macd and signal line
  #calculate short term exonential moving average (ema)
  ShortEMA = df.High.ewm(span=12, adjust=False).mean()#short term last 12periods usually
  #calculate longer term exponential moving average
  LongEMA= df.High.ewm(span=26, adjust=False).mean()#long term last 26periods
  #calculate the macd line
  MACD = ShortEMA - LongEMA
  #Calculate Signal LIne
  signal = MACD.ewm(span=9, adjust=False).mean()

  df['MACD'] = MACD
  df['Signal Line'] = signal
  #print(df)

  #create buy and sell column
  a = buy_sellmacd(df)
  df['Buy'] = a[0]
  df['Sell'] = a[1]
  macdPlot(df)

  print(df.tail(5)[['Close','Buy', 'Sell']])

def rsidata(df):
    delta = df['Close'].diff(1) #difference in price from previous day
    delta = delta.dropna()
    up = delta.copy()
    down = delta.copy() 

    up[up<0] = 0 
    down[down>0] = 0
    period = 14
    #calculate the average gain and average loss
    avg_gain = up.rolling(window=period).mean()
    avg_loss=abs(down.rolling(window=period).mean())
    rs = avg_gain/avg_loss

    #calculate relative strength index for rsi 
    rsi = 100.0 - (100.0/ (1.0 + rs))

    new_df = pd.DataFrame()
    new_df['Close'] = df['Close']
    new_df['RSI'] = rsi


  #visually show the adj close price and rsi 
# plot the adj close price 

 
  #plot the correspodning rsi values and significant level
    plt.figure(figsize=(12.6, 4.6))
    plt.title('RSI Plot')
    plt.plot(new_df['RSI'])
    plt.axhline(0, linestyle='-', alpha = 0.5, color ='gray')
    plt.axhline(10, linestyle='-', alpha = 0.5, color ='orange')
    plt.axhline(20, linestyle='-', alpha = 0.5, color ='green')
    plt.axhline(30, linestyle='-', alpha = 0.5, color ='red')
    plt.axhline(70, linestyle='-', alpha = 0.5, color ='red')
    plt.axhline(80, linestyle='-', alpha = 0.5, color ='green')
    plt.axhline(90, linestyle='-', alpha = 0.5, color ='orange')
    plt.axhline(100, linestyle='-', alpha = 0.5, color ='gray')
    plt.show()

def buy_sell_function(data):
      
  buy_list=[]
  sell_list=[]
  flag_long = False
  flag_short = False

  for i in range(0, len(data)):
    if data['Middle'][i] < data['Long'][i] and data['Short'][i] < data['Middle'][i] and flag_long == False and flag_short == False:
      buy_list.append(data['Close'][i]) 
      sell_list.append(np.nan)
      flag_short = True
    elif flag_short == True and data['Short'][i] > data['Middle'][i]:
      sell_list.append(data['Close'][i])
      buy_list.append(np.nan)
      flag_short = False
    elif data['Middle'][i] > data['Long'][i] and data['Short'][i] > data['Middle'][i] and flag_long == False and flag_short == False:
      buy_list.append(data['Close'][i]) 
      sell_list.append(np.nan)
      flag_long = True
    elif flag_long == True and data['Short'][i] < data['Middle'][i]:
      sell_list.append(data['Close'][i])
      buy_list.append(np.nan)
      flag_long = False
    else: 
      buy_list.append(np.nan)
      sell_list.append(np.nan)
  
  return(buy_list, sell_list)

def threema(df): 
  ShortEMA = df.Close.ewm(span=5, adjust = False).mean()
  #calculate the middle/medium exponential moving average
  MiddleEMA = df.Close.ewm(span=34, adjust = False).mean() #change span to 21
  #calculate the long/slow exponential moving average 
  LongEMA= df.Close.ewm(span=63, adjust = False).mean() 

  df['Short'] = ShortEMA
  df['Middle'] = MiddleEMA
  df['Long'] = LongEMA
  df['Buy'] = buy_sell_function(df)[0]
  df['Sell'] = buy_sell_function(df)[1]


  plt.figure(figsize=(12.2, 4.5))
  plt.title('Three Moving Averages')
  plt.plot(df['Close'], color = 'blue', alpha = 0.35)
  plt.plot(ShortEMA, label = 'Short EMA', color = 'red', alpha = 0.35)
  plt.plot(MiddleEMA, label = 'Middle EMA', color = 'orange', alpha = 0.35)
  plt.plot(LongEMA, label = 'Long EMA', color = 'green', alpha = 0.35)
  plt.scatter(df.index, df['Buy'], color = 'green', marker = '^', alpha = 1)
  plt.scatter(df.index, df['Sell'], color = 'red', marker = 'v', alpha = 1)
  plt.xlabel('Date', fontsize = 18)
  plt.ylabel('Close Price', fontsize = 18)
  plt.legend(loc = 'lower left')
  plt.show()

  print(df.tail(5)[['Close','Buy', 'Sell']])
  
  
today = datetime.today()
enddate = today + timedelta(1)
enddate = enddate.strftime('%Y-%m-%d')
#ticks = ["CMMB","OCGN","CLOV","CLOVW", "PLBY", "NCTY", "AEMD", "AMC", "NKLA", "W", "CZR", "GME", "ATER", "WISH", "GEO", "AEI", "AHT", "BTCM", "MDLY", "QS", "OPENW", "CLNE", "PUBM", "HOOK", "FCEL", "TAL", "ROOT", "TIL", "SOL", "BTC-USD", "ETH-USD", "DOGE-USD"]
ticks = ["BIOC"]# "IDEX", "PBTS", "ALRN", "QD", "BIOL", "CLNE", "MRNS", "ATHE", "MICT", "TEDU", "F", "OSS", "AIHS", "BRQS", "BBI", "ANIX", "NAK", "HJLI", "NAOV", "GME", "AMC", "BTC-USD", "ETH-USD", "DOGE-USD", "NOK", "PFE"]
#"KTOV", "PBTS", "ALRN", "CLNE", "MRNS", "ATHE", "MICT", "TEDU", "F", "OSS", "AIHS", "BRQS", "BBI", "ANIX", "NAK", "HJLI", "NAOV", "GME", "AMC", "BTC-USD", "ETH-USD", "DOGE-USD", "NOK", "PFE"



for i in ticks:
  tickerDf = yf.download(i, interval='1d', start='2019-5-1', end=enddate)
  df = pd.DataFrame(tickerDf)
  print(i)
  print()
  plotog(df)
  print()
  dma(df)
  print()
  fnMacd(df)
  print()
  threema(df)
  print()
  rsidata(df)

