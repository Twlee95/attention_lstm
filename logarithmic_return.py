import pandas_datareader.data as pdr
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import LSTM_MODEL_DATASET as LSTMMD
import torch

import sys
for i in sys.path:
    print(i)

sys.path.append('C:\\Users\\leete\\PycharmProjects\\LSTM')
sys.path.remove('C:\\Users\\lee\\PycharmProjects\\LSTM')
os.getcwd()
os.chdir('C:\\Users\\lee\\PycharmProjects')
os.chdir('C:\\Users\\lee\\PycharmProjects\\LSTM')

from scipy.stats import skew, kurtosis
## statistics
def statistics(data):
    mean = sum(data['Close'])/len(data['Close'])
    sd = np.std(data['Close'])
    ske = skew(data['Close'])
    kur = kurtosis(data['Close'], fisher=False)
    print('mean:{0:3.3f}, sd:{1:3.3f}, skew:{2:3.3f},kurto :{3:3.3f}'.format(mean,sd,ske,kur))

start = datetime.datetime(*(2000, 1, 1))
end = datetime.datetime(*(2020, 12, 31))


# '^KS11' : KOSPI
KOSPI_data = pdr.DataReader('^KS11','yahoo', start, end)
KOSPI_df = pd.DataFrame(KOSPI_data.Close)
trainset = LSTMMD.StockDataset('^KS11', 7, 1, (2000, 1, 1), (2012, 12, 31))
valset = LSTMMD.StockDataset('^KS11', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('^KS11', 7, 1, (2017, 1, 1), (2020, 12, 31))
print(len(KOSPI_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(KOSPI_data)

from statsmodels.stats.stattools import jarque_bera
print(jarque_bera(KOSPI_df['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(KOSPI_df['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# '^KQ11' : 코스닥
KOSDAQ_data = pdr.DataReader('^KQ11','yahoo', datetime.datetime(*(2013, 3, 3)), end)
KOSDAQ_df = pd.DataFrame(KOSDAQ_data.Close)
trainset = LSTMMD.StockDataset('^KQ11', 7, 1, (2013, 3, 3), (2016, 12, 31))
valset = LSTMMD.StockDataset('^KQ11', 7, 1, (2017, 1, 1), (2018, 12, 31))
testset = LSTMMD.StockDataset('^KQ11', 7, 1, (2019, 1, 1), (2020, 12, 31))
print(len(KOSDAQ_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(KOSDAQ_data)

from statsmodels.stats.stattools import jarque_bera
print(jarque_bera(KOSDAQ_df['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(KOSDAQ_df['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))



# '^IXIC' : 나스닥
NASDAQ_data = pdr.DataReader('^IXIC','yahoo', start, end)
NASDAQ_df = pd.DataFrame(NASDAQ_data.Close)
trainset = LSTMMD.StockDataset('^IXIC', 7, 1, (2000, 1, 1), (2012, 12, 31))
valset = LSTMMD.StockDataset('^IXIC', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('^IXIC', 7, 1, (2017, 1, 1), (2020, 12, 31))
print(len(NASDAQ_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(NASDAQ_data)
print(jarque_bera(NASDAQ_data['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(NASDAQ_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# '^GSPC' : SNP 500 지수
SNP_data = pdr.DataReader('^GSPC','yahoo', start, end)
df = SNP_data['Close']
SNP_df = pd.DataFrame(SNP_data.Close)
trainset = LSTMMD.StockDataset('^GSPC', 7, 1, (2000, 1, 1), (2012, 12, 31))
valset = LSTMMD.StockDataset('^GSPC', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('^GSPC', 7, 1, (2017, 1, 1), (2020, 12, 31))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose

print(df)

df.dropna(inplace=True)
plt.plot(df)

result = seasonal_decompose(df, model='additive',period=240)
plt.rcParams['figure.figsize'] = [12, 8]
result.plot()
plt.show()



print(len(SNP_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(SNP_data)
print(jarque_bera(SNP_data['Close']))
from statsmodels.tsa.stattools import adfuller
result = adfuller(SNP_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# '^DJI' : 다우존수 산업지수
daw_data = pdr.DataReader('^DJI','yahoo', start, end)
daw_df = pd.DataFrame(daw_data.Close)
trainset = LSTMMD.StockDataset('^DJI', 7, 1, (2000, 1, 1), (2012, 12, 31))
valset = LSTMMD.StockDataset('^DJI', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('^DJI', 7, 1, (2017, 1, 1), (2020, 12, 31))
print(len(daw_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(daw_data)
print(jarque_bera(daw_data['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(daw_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# '^HSI' : 홍콩 항생 지수
han_data = pdr.DataReader('^HSI','yahoo', start, end)
han_df = pd.DataFrame(han_data.Close)
trainset = LSTMMD.StockDataset('^HSI', 7, 1, (2000, 1, 1), (2012, 12, 31))
valset = LSTMMD.StockDataset('^HSI', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('^HSI', 7, 1, (2017, 1, 1), (2020, 12, 31))
print(len(han_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(han_data)
print(jarque_bera(han_data['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(han_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# '^N225' : 니케이지수
nikei_data = pdr.DataReader('^N225','yahoo', start, end)
nikei_df = pd.DataFrame(nikei_data.Close)
trainset = LSTMMD.StockDataset('^N225', 7, 1, (2000, 1, 1), (2012, 12, 31))
valset = LSTMMD.StockDataset('^N225', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('^N225', 7, 1, (2017, 1, 1), (2020, 12, 31))
print(len(nikei_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(nikei_data)
print(jarque_bera(nikei_data['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(nikei_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# '^GDAXI' : 독일 DAX
dax_data = pdr.DataReader('^GDAXI','yahoo', start, end)
dax_df = pd.DataFrame(dax_data.Close)
trainset = LSTMMD.StockDataset('^GDAXI', 7, 1, (2000, 1, 1), (2012, 12, 31))
valset = LSTMMD.StockDataset('^GDAXI', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('^GDAXI', 7, 1, (2017, 1, 1), (2020, 12, 31))
print(len(dax_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(dax_data)
print(jarque_bera(dax_data['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(dax_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# '^IBEX' : 스페인 IBEX
SP_IBEX_data = pdr.DataReader('^IBEX','yahoo', datetime.datetime(*(2000, 1, 1)), end)
SP_IBEX_df = pd.DataFrame(SP_IBEX_data.Close)
trainset = LSTMMD.StockDataset('^IBEX', 7, 1, (2000, 1, 1), (2012, 12, 31))
valset = LSTMMD.StockDataset('^IBEX', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('^IBEX', 7, 1, (2017, 1, 1), (2020, 12, 31))
print(len(SP_IBEX_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(SP_IBEX_data)

from statsmodels.stats.stattools import jarque_bera
print(jarque_bera(SP_IBEX_df['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(SP_IBEX_df['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# 'FTSEMIB.MI' : 이탈리아 FTSE MIB
ITLY_data = pdr.DataReader('FTSEMIB.MI','yahoo', datetime.datetime(*(2000, 1, 1)), end)
ITLY_df = pd.DataFrame(ITLY_data.Close)
trainset = LSTMMD.StockDataset('FTSEMIB.MI', 7, 1, (2000, 1, 1), (2012, 12, 31))
valset = LSTMMD.StockDataset('FTSEMIB.MI', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('FTSEMIB.MI', 7, 1, (2017, 1, 1), (2020, 12, 31))
print(len(ITLY_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(ITLY_data)

from statsmodels.stats.stattools import jarque_bera
print(jarque_bera(ITLY_df['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(ITLY_df['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# '^FTSE' : 영국 FTSE  ############################3
eng_data = pdr.DataReader('^FTSE','yahoo', start, end)
eng_df = pd.DataFrame(eng_data.Close)
trainset = LSTMMD.StockDataset('^FTSE', 7, 1, (2000, 1, 1), (2012, 12, 31))
valset = LSTMMD.StockDataset('^FTSE', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('^FTSE', 7, 1, (2017, 1, 1), (2020, 12, 31))
print(len(eng_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(eng_data)
print(jarque_bera(eng_data['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(eng_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# '^FCHI' : 프랑스 CAC
cac_data = pdr.DataReader('^FCHI','yahoo', start, end)
cac_df = pd.DataFrame(cac_data.Close)
trainset = LSTMMD.StockDataset('^FCHI', 7, 1, (2000, 1, 1), (2012, 12, 31))
valset = LSTMMD.StockDataset('^FCHI', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('^FCHI', 7, 1, (2017, 1, 1), (2020, 12, 31))
print(len(cac_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(cac_data)
print(jarque_bera(cac_data['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(cac_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# '^TWII' : 대만 기권
thai_data = pdr.DataReader('^TWII','yahoo', start, end)
thai_df = pd.DataFrame(thai_data.Close)
trainset = LSTMMD.StockDataset('^TWII', 7, 1, (2000, 1, 1), (2012, 12, 31))
valset = LSTMMD.StockDataset('^TWII', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('^TWII', 7, 1, (2017, 1, 1), (2020, 12, 31))
print(len(thai_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(thai_data)
print(jarque_bera(thai_data['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(thai_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))



# '^AEX' : 네덜란드 AEX
ned_data = pdr.DataReader('^AEX','yahoo', start, end)
ned_df = pd.DataFrame(ned_data.Close)
trainset = LSTMMD.StockDataset('^AEX', 7, 1, (2000, 1, 1), (2012, 12, 31))
valset = LSTMMD.StockDataset('^AEX', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('^AEX', 7, 1, (2017, 1, 1), (2020, 12, 31))
print(len(ned_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(ned_data)
print(jarque_bera(ned_data['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(ned_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# '^BSESN' : 인도 센섹스
india_data = pdr.DataReader('^BSESN','yahoo', start, end)
india_df = pd.DataFrame(india_data.Close)
trainset = LSTMMD.StockDataset('^BSESN', 7, 1, (2000, 1, 1), (2012, 12, 31))
valset = LSTMMD.StockDataset('^BSESN', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('^BSESN', 7, 1, (2017, 1, 1), (2020, 12, 31))
print(len(india_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(india_data)
print(jarque_bera(india_data['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(india_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))



# 'RTSI.ME' : 러시아 RTXI
russia_data = pdr.DataReader('RTSI.ME','yahoo', start, end)
russia_df = pd.DataFrame(russia_data.Close)
trainset = LSTMMD.StockDataset('RTSI.ME', 7, 1, (2000, 1, 1), (2012, 12, 31))
valset = LSTMMD.StockDataset('RTSI.ME', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('RTSI.ME', 7, 1, (2017, 1, 1), (2020, 12, 31))
print(len(russia_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(russia_data)
print(jarque_bera(russia_data['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(russia_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# '^BVSP' : 브라질 보베스파 지수
brazil_data = pdr.DataReader('^IXIC','yahoo', start, end)
brazil_df = pd.DataFrame(brazil_data.Close)
trainset = LSTMMD.StockDataset('^IXIC', 7, 1, (2000, 1, 1), (2012, 12, 31))
valset = LSTMMD.StockDataset('^IXIC', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('^IXIC', 7, 1, (2017, 1, 1), (2020, 12, 31))
print(len(NASDAQ_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(brazil_data)
print(jarque_bera(brazil_data['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(brazil_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# 'GC=F' : 금 가격
gold_data = pdr.DataReader('GC=F','yahoo', start, end)
gold_df = pd.DataFrame(gold_data.Close)
trainset = LSTMMD.StockDataset('GC=F', 7, 1, (2000, 1, 1), (2012, 12, 31))
valset = LSTMMD.StockDataset('GC=F', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('GC=F', 7, 1, (2017, 1, 1), (2020, 12, 31))
print(len(gold_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(gold_data)
print(jarque_bera(gold_data['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(gold_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# 'CL=F' : 원유 가격 (2000/ 8 / 20일 부터 데이터가 있음)
oil_data = pdr.DataReader('CL=F','yahoo', datetime.datetime(*(2000, 8, 23)), end)
oil_df = pd.DataFrame(oil_data.Close)
trainset = LSTMMD.StockDataset('CL=F', 7, 1, (2000, 8, 23), (2012, 12, 31))
valset = LSTMMD.StockDataset('CL=F', 7, 1, (2013, 1, 1), (2016, 12, 31))
testset = LSTMMD.StockDataset('CL=F', 7, 1, (2017, 1, 1), (2020, 12, 31))
print(len(oil_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))
statistics(oil_data)
print(jarque_bera(oil_data['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(oil_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# 10년만기 미국 국채


# 'BTC-USD' : 비트코인 암호화폐
def c_statistics(data):
    mean = sum(data)/len(data)
    sd = np.std(data)
    ske = skew(data)
    kur = kurtosis(data, fisher=False)
    print('mean:{0:3.3f}, sd:{1:3.3f}, skew:{2:3.3f},kurto :{3:3.3f}'.format(mean,sd,ske,kur))


bcoin_data = pdr.DataReader('BTC-USD','yahoo', datetime.datetime(*(2014, 9, 17)), end)
bcoin_df = pd.DataFrame(bcoin_data.Close)
trainset = LSTMMD.StockDataset('BTC-USD', 7, 1, (2014, 9 , 17), (2018, 12, 31))
valset = LSTMMD.StockDataset('BTC-USD', 7, 1, (2019, 1, 1), (2019, 12, 31))
testset = LSTMMD.StockDataset('BTC-USD', 7, 1, (2020, 1, 1), (2020, 12, 31))
print(len(bcoin_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))

data = bcoin_data.astype({'Close': 'float'})
data = data.loc[:, 'Close']
c_statistics(data)
print(jarque_bera(bcoin_data['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(bcoin_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# 'ETH-USD' : 이더리움 암호화폐
ecoin_data = pdr.DataReader('ETH-USD','yahoo', datetime.datetime(*(2015, 8, 7)), end)
ecoin_df = pd.DataFrame(ecoin_data.Close)
trainset = LSTMMD.StockDataset('ETH-USD', 7, 1, (2015, 8, 7), (2018, 12, 31))
valset = LSTMMD.StockDataset('ETH-USD', 7, 1, (2019, 1, 1), (2019, 12, 31))
testset = LSTMMD.StockDataset('ETH-USD', 7, 1, (2020, 1, 1), (2020, 12, 31))
print(len(ecoin_df))
print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))

data = ecoin_data.astype({'Close': 'float'})
data = data.loc[:, 'Close']
c_statistics(data)
print(jarque_bera(ecoin_data['Close']))

from statsmodels.tsa.stattools import adfuller
result = adfuller(ecoin_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))




import csv

f = open("C:\\Users\\lee\\PycharmProjects\\LSTM\\data\\10년만기 미국채 선물 역사적 데이터.csv", 'r', encoding='utf-8')
rdr = csv.reader(f)
data = []
for line in rdr:
    print(line)
    data.append(line)
f.close()

data = pd.DataFrame(data)

bond_colname = data.iloc[0, :]
bond_colname2 = bond_colname.iloc[1:]
bond_colname2 = list(bond_colname2)

new_bond_df = data.iloc[1:,:]


new_bond_df2 = new_bond_df.sort_values(by=0, axis=0, ascending=True)

bond_date = new_bond_df2.iloc[:, 0]
bond_date= list(bond_date)
data_ind = pd.DatetimeIndex(bond_date)

new_bond_df3 = new_bond_df2.iloc[:,1:7]

new_bond_df3


df = pd.DataFrame(np.array(new_bond_df3), index = data_ind, columns= bond_colname2)
ddf =df.iloc[0:11]

ddf[['종가']]

df
df.to_csv('C:\\Users\\lee\\PycharmProjects\\LSTM\\data\\us_10y_tb.csv', sep=',', na_rep='NaN')


# KOSPI200_data = pdr.DataReader('^KOSPI200' ,'yahoo', start, end)
# KOSPI200_df = pd.DataFrame(KOSPI200_data.Close)
# len()
NASDAQ_data = pdr.DataReader('^IXIC','yahoo', start, end)
NASDAQ_df = pd.DataFrame(NASDAQ_data.Close)
len(NASDAQ_df)
gold_data = pdr.DataReader('GC=F','yahoo', start, end)
gold_df = pd.DataFrame(gold_data.Close)
len(gold_df)
oil_data = pdr.DataReader('CL=F','yahoo', start, end)
oil_df = pd.DataFrame(oil_data.Close)
len(oil_df)

KOSPI_df['log((t+1)/t)'] = np.log(KOSPI_df.Close) - np.log(KOSPI_df.Close.shift(1))
len(KOSPI_df['log((t+1)/t)'])

plt.plot(KOSPI_df['log((t+1)/t)'],lw=0.4)
plt.title('KOSPI_logarithmic return')
plt.xlabel('time')
plt.ylabel('return')


NASDAQ_df['log((t+1)/t)'] = np.log(NASDAQ_df.Close) - np.log(NASDAQ_df.Close.shift(1))
len(NASDAQ_df['log((t+1)/t)'])

plt.plot(NASDAQ_df['log((t+1)/t)'],lw=0.4)
plt.title('NASDAQ_logarithmic return')
plt.xlabel('time')
plt.ylabel('return')

gold_df['log((t+1)/t)'] = np.log(gold_df.Close) - np.log(gold_df.Close.shift(1))
len(gold_df['log((t+1)/t)'])

plt.plot(gold_df['log((t+1)/t)'],lw=0.4)
plt.title('gold_logarithmic return')
plt.xlabel('time')
plt.ylabel('return')

oil_df['log((t+1)/t)'] = np.log(oil_df.Close) - np.log(oil_df.Close.shift(1))
len(oil_df['log((t+1)/t)'])

plt.plot(oil_df['log((t+1)/t)'],lw=0.4)
plt.title('oil_logarithmic return')
plt.xlabel('time')
plt.ylabel('return')


plt.hist(KOSPI_df['log((t+1)/t)'])
plt.title('KOSPI_logarithmic return')
plt.xlabel('time')
plt.ylabel('return')

plt.hist(NASDAQ_df['log((t+1)/t)'])
plt.hist(gold_df['log((t+1)/t)'])
plt.hist(oil_df['log((t+1)/t)'])



trainset[0]

data = LSTMMD.StockDataset('GC=F', 10, 1, (2000, 1, 1), (2020, 12, 31))

data_list = []
for i in range(len(data)):
    data_list.append(data[i][1][0][0])

df = pd.DataFrame(data_list)

parameters = {'xtick.labelsize': 15,
          'ytick.labelsize': 15}
plt.rcParams.update(parameters)

plt.plot(df)

plt.plot(df[["train_losses"]].values[0][0])
plt.plot(df[["val_losses"]].values[0][0])
plt.legend(['train_losses', 'val_losses'],fontsize=15)
plt.xlabel('epoch',fontsize=15)
plt.ylabel('loss',fontsize=15)
plt.grid()



print(len(trainset)+len(valset)+len(testset))
print(len(trainset))
print(len(valset))
print(len(testset))






from statsmodels.stats.stattools import jarque_bera

from statsmodels.tsa.stattools import adfuller
result = adfuller(KOSPI_df['log((t+1)/t)'][1:])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

def statistics2(data):
    mean = sum(data['log((t+1)/t)'])/len(data['log((t+1)/t)'])
    sd = np.std(data['log((t+1)/t)'])
    ske = skew(data['log((t+1)/t)'])
    kur = kurtosis(data['log((t+1)/t)'], fisher=False)
    print('mean:{0:3.3f}, sd:{1:3.3f}, skew:{2:3.3f},kurto :{3:3.3f}'.format(mean,sd,ske,kur))

statistics2(KOSPI_df['log((t+1)/t)'][1:])






sum(KOSPI_df['log((t+1)/t)'][1:])/len(KOSPI_df['log((t+1)/t)'][1:])
np.std(KOSPI_df['log((t+1)/t)'][1:])
skew(KOSPI_df['log((t+1)/t)'][1:])
kurtosis(KOSPI_df['log((t+1)/t)'][1:], fisher=False)


