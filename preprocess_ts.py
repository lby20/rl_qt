import pandas as pd
import talib


df =  pd.read_csv('.data/IC_2015to2018.csv',parse_dates=True,index_col=0) 

df['return'] = df['close']-df['open']/df['open']



## techinical index
df['MACD']=talib.MACD(df.close, fastperiod=12, slowperiod=26, signalperiod=9)[2]
df['EMA_7']= talib.EMA(df.close,timeperiod=7)
df['EMA_21']=talib.EMA(df.close,timeperiod=21)
df['EMA_56']=talib.EMA(df.close,timeperiod=56)
df['RSI']=talib.RSI(df.close,timeperiod=56)
df['BB_up']=talib.BBANDS(df.close)[0]
df['BB_mid']=talib.BBANDS(df.close)[1]
df['BB_low']=talib.BBANDS(df.close)[2]
df['slowK']=talib.STOCH(df.high,df.low,df.close)[0]
df['slowD']=talib.STOCH(df.high,df.low,df.close)[1]


df.dropna(inplace=True)
# df.to_csv('IF_tech.csv') #相對位置，保存在getwcd()獲得的路徑下
df.to_csv('IC_2015to2018_tech.csv') #相對位置，保存在getwcd()獲得的路徑下