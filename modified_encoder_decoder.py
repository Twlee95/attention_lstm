
import sys
sys.path.append('C:\\Users\\lee\\PycharmProjects\\LSTM')
import time
import pandas as pd
import pandas_datareader.data as pdr
import datetime
import pylab
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import argparse
from copy import deepcopy # Add Deepcopy for args
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns
import matplotlib.pyplot as plt
import LSTM_MODEL_DATASET as LSTMMD
import csv
import os






class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, batch_size, num_layers=1 ):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        # define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)

        self.hidden = self.init_hidden()
    def init_hidden(self):
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        '''

        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))



    def forward(self, x_input):
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        '''

        lstm_out, self.hidden = self.lstm(x_input, self.hidden)

        return lstm_out, self.hidden

class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers=1):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers)
        self.regressor = self.make_regressor()

    def make_regressor(self):  # 간단한 MLP를 만드는 함수
        layers = []
        # if self.use_bn:
        #     layers.append(nn.BatchNorm1d(self.hidden_dim))  ##  nn.BatchNorm1d
        # layers.append(nn.Dropout(self.dropout))    ##  nn.Dropout

        ## hidden dim을 outputdim으로 바꿔주는 MLP
        # layers.append(nn.Linear(self.hidden_dim, self.hidden_dim // 2))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(self.hidden_dim // 2, self.output_dim)) # 여기서 output dim이 사용됨
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        regressor = nn.Sequential(*layers)
        return regressor

    def forward(self, x_input, encoder_hidden_states):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''
        lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
        y_pred = self.regressor(lstm_out[-1].view(self.batch_size, -1))

        return y_pred, self.hidden





def metric(y_pred, y_true):
    perc_y_pred = np.exp(y_pred.cpu().detach().numpy())
    perc_y_true = np.exp(y_true.cpu().detach().numpy())
    # mean_absolute_error : 차이의 절댓값을 loss function으로 사용
    mae = mean_absolute_error(perc_y_true, perc_y_pred, multioutput='raw_values')
    return mae

def metric2(y_pred, y_true):
    perc_y_pred = np.exp(y_pred.cpu().detach().numpy())
    #print('perc_y_pred :{},perc_y_pred.shape :{}'.format(perc_y_pred,perc_y_pred.shape))
    perc_y_true = np.exp(y_true.cpu().detach().numpy())
    # mean_absolute_error : 차이의 절댓값을 loss function으로 사용
    mse = mean_squared_error(perc_y_true, perc_y_pred, multioutput='raw_values')
    # y_pred = np.array(y_pred)
    # y_true = np.array(y_true)
    # mse = mean_squared_error(y_pred, y_true, multioutput='raw_values')
    return mse

def metric3(y_pred, y_true):
    perc_y_pred = np.exp(y_pred.cpu().detach().numpy())
    perc_y_true = np.exp(y_true.cpu().detach().numpy())
    # mean_absolute_error : 차이의 절댓값을 loss function으로 사용
    mape = mean_absolute_percentage_error(perc_y_true, perc_y_pred, multioutput='raw_values')
    return mape

def train(encoder,decoder, partition, enc_optimizer,dec_optimizer, loss_fn, args):
    trainloader = DataLoader(partition['train'],    ## DataLoader는 dataset에서 불러온 값으로 랜덤으로 배치를 만들어줌
                             batch_size=args.batch_size,
                             shuffle=True, drop_last=True)
    encoder.train()
    decoder.train()
    encoder.zero_grad()
    decoder.zero_grad()
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()

    train_loss = 0.0
    for X,y in trainloader:
        X = X.transpose(0, 1).float().to(args.device)
        encoder.zero_grad()
        enc_optimizer.zero_grad()
        encoder.hidden = [hidden.to(args.device) for hidden in encoder.init_hidden()]
        y_pred_enc, encoder_hidden = encoder(X)

        decoder_hidden = encoder_hidden

        y_true = y[:, :].float().to(args.device)
        decoder.zero_grad()
        dec_optimizer.zero_grad()
        # decoder.hidden = [hidden.to(args.device) for hidden in decoder.init_hidden()]
        y_pred_dec, decoder_hidden = decoder(X, decoder_hidden)

        loss = loss_fn(y_pred_dec.view(-1), y_true.view(-1))  # .view(-1)은 1열로 줄세운것
        loss.backward()  ## gradient 계산

        enc_optimizer.step()  ## parameter 갱신
        dec_optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / len(trainloader)
    return encoder, decoder, train_loss


def validate(encoder, decoder, partition, loss_fn, args):
    valloader = DataLoader(partition['val'],
                           batch_size=args.batch_size,
                           shuffle=False, drop_last=True)
    encoder.eval()
    decoder.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X, y in valloader:

            X = X.transpose(0, 1).float().to(args.device)
            encoder.hidden = [hidden.to(args.device) for hidden in encoder.init_hidden()]
            y_pred_enc, encoder_hidden = encoder(X)

            decoder_hidden = encoder_hidden

            y_true = y[:, :].float().to(args.device)
            # decoder.hidden = [hidden.to(args.device) for hidden in decoder.init_hidden()]

            y_pred_dec, decoder_hidden = decoder(X,decoder_hidden)
            # print('validate y_pred: {}, y_pred.shape : {}'. format(y_pred, y_pred.shape))
            loss = loss_fn(y_pred_dec.view(-1), y_true.view(-1))

            val_loss += loss.item()

    val_loss = val_loss / len(valloader)
    return val_loss

def test(encoder,decoder, partition, args):
    testloader = DataLoader(partition['test'],
                           batch_size=args.batch_size,
                           shuffle=False, drop_last=True)
    encoder.eval()
    decoder.eval()
    test_loss_metric = 0.0
    test_loss_metric2 = 0.0
    test_loss_metric3 = 0.0
    with torch.no_grad():
        for X, y in testloader:
            X = X.transpose(0, 1).float().to(args.device)
            encoder.hidden = [hidden.to(args.device) for hidden in encoder.init_hidden()]
            y_pred_enc, encoder_hidden = encoder(X)

            decoder_hidden = encoder_hidden

            y_true = y[:, :].float().to(args.device)
            # decoder.hidden = [hidden.to(args.device) for hidden in decoder.init_hidden()]

            y_pred_dec , decoder_hidden= decoder(X,decoder_hidden)
            # print('test y_pred: {},shape : {}'.format(y_pred, y_pred.shape))
            # print('test y_pred: {},shape : {}'.format(y_true, y_true.shape))
            # print('test metric(y_pred, y_true): {},shape : {}'.format(metric(y_pred, y_true), metric(y_pred, y_true).shape))
            # print('test metric(y_pred, y_true)[0]: {},shape : {}'.format(metric(y_pred, y_true)[0] ,metric(y_pred, y_true)[0].shape))

            test_loss_metric += args.metric(y_pred_dec, y_true.squeeze())[0]
            test_loss_metric2 += args.metric2(y_pred_dec, y_true.squeeze())[0]
            test_loss_metric3 += args.metric3(y_pred_dec, y_true.squeeze())[0]

    test_loss_metric = test_loss_metric / len(testloader)
    test_loss_metric2 = test_loss_metric2 / len(testloader)
    test_loss_metric3 = test_loss_metric3 / len(testloader)
    return test_loss_metric, test_loss_metric2, test_loss_metric3


def experiment(partition, args):
    encoder = args.encoder(args.input_dim, args.hid_dim, args.batch_size, args.n_layers)
    decoder = args.decoder(args.input_dim, args.hid_dim, args.batch_size, args.y_frames, args.n_layers )
    encoder.to(args.device)
    decoder.to(args.device)

    loss_fn = nn.MSELoss()
    # loss_fn.to(args.device) ## gpu로 보내줌  간혹 loss에 따라 안되는 경우도 있음
    if args.optim == 'SGD':
        enc_optimizer = optim.RMSprop(encoder.parameters(), lr=args.lr, weight_decay=args.l2)
        dec_optimizer = optim.RMSprop(decoder.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'RMSprop':
        enc_optimizer = optim.RMSprop(encoder.parameters(), lr=args.lr, weight_decay=args.l2)
        dec_optimizer = optim.RMSprop(decoder.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'Adam':
        enc_optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.l2)
        dec_optimizer = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise ValueError('In-valid optimizer choice')

    # ===== List for epoch-wise data ====== #
    train_losses = []
    val_losses = []
    # ===================================== #

    for epoch in range(args.epoch):  # loop over the dataset multiple times
        ts = time.time()
        encoder, decoder, train_loss= train(encoder, decoder, partition, enc_optimizer, dec_optimizer, loss_fn, args)
        val_loss = validate(encoder, decoder, partition, loss_fn, args)
        te = time.time()

        # ====== Add Epoch Data ====== #
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # ============================ #
        torch.save(encoder.state_dict(), args.innate_path + '\\' + str(epoch) + '_epoch'+'_Encoder'+ '.pt')
        torch.save(decoder.state_dict(), args.innate_path + '\\' + str(epoch) + '_epoch'+'_Decoder' + '.pt')
        print('Epoch {}, Loss(train/val) {:2.5f}/{:2.5f}. Took {:2.2f} sec'.format(epoch,train_loss,val_loss,te - ts))

    site_val_losses = val_losses.index(min(val_losses)) ## 10 epoch일 경우 0번째~9번째 까지로 나옴
    encoder = args.encoder(args.input_dim, args.hid_dim, args.batch_size, args.n_layers)
    decoder = args.decoder(args.input_dim, args.hid_dim, args.batch_size, args.y_frames, args.n_layers)
    encoder.to(args.device)
    decoder.to(args.device)
    encoder.load_state_dict(torch.load(args.innate_path + '\\' + str(site_val_losses) +'_epoch' +'_Encoder'+ '.pt'))
    decoder.load_state_dict(torch.load(args.innate_path + '\\' + str(site_val_losses) + '_epoch'+'_Decoder' + '.pt'))

    test_loss_metric, test_loss_metric2, test_loss_metric3 = test(encoder, decoder, partition, args)
    print('test_loss_metric: {}, test_loss_metric2: {},test_loss_metric3: {}'.format(test_loss_metric,test_loss_metric2,test_loss_metric3))

    with open(args.innate_path + '\\'+ str(site_val_losses)+'Epoch_test_metric' +'.txt', 'w') as f:
        print('test_loss_metric : {} \n test_loss_metric2 : {} \n test_loss_metric3 : {}'.format(test_loss_metric, test_loss_metric2, test_loss_metric3), file=f)
    # ======= Add Result to Dictionary ======= #
    result = {}
    result['train_losses'] = train_losses
    result['val_losses'] = val_losses
    result['test_loss_metric'] = test_loss_metric
    result['test_loss_metric2'] = test_loss_metric2
    result['test_loss_metric3'] = test_loss_metric3
    return vars(args), result




# manage experiment
import hashlib
import json
from os import listdir
from os.path import isfile, join
import pandas as pd


def save_exp_result(setting, result):
    exp_name = setting['exp_name']
    del setting['epoch']

    hash_key = hashlib.sha1(str(setting).encode()).hexdigest()[:6]
    filename = './results/{}-{}.json'.format(exp_name, hash_key)
    result.update(setting)
    with open(filename, 'w') as f:
        json.dump(result, f)


def load_exp_result(exp_name):
    dir_path = 'results'
    filenames = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) if '.json' in f]
    list_result = []
    for filename in filenames:
        if exp_name in filename:
            with open(join(dir_path, filename), 'r') as infile:
                results = json.load(infile)
                list_result.append(results)
    df = pd.DataFrame(list_result)  # .drop(columns=[])
    return df

# ====== Random Seed Initialization ====== #
seed = 666
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args("")
args.exp_name = "exp1_lr"
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
# '^KS11' : KOSPI
# '^KS200' : KOSPI 200
# '^IXIC' : 나스닥
# 'GC=F' : 금 가격
# 'CL=F' : 원유 가격
# ====== Data Loading ====== #
args.symbol = '^IBEX'
args.data_site = 'C:\\Users\\lee\\PycharmProjects\\LSTM\\data\\us_10y_tb.csv'
args.batch_size = 128
args.x_frames = 7
args.y_frames = 1
args.model = LSTMMD.GRU
args.encoder = lstm_encoder
args.decoder = lstm_decoder

# ====== Model Capacity ===== #
args.input_dim = 1
args.hid_dim = 10
args.n_layers = 1

# ====== Regularization ======= #
args.l2 = 0.00001
args.dropout = 0.0
args.use_bn = True

# ====== Optimizer & Training ====== #
args.optim = 'RMSprop'  # 'RMSprop' #SGD, RMSprop, ADAM...
args.lr = 0.0001
args.epoch = 2

# ====== Experiment Variable ====== #
name_var1 = 'lr'
name_var2 = 'n_layers'
list_var1 = [0.001, 0.0001, 0.00001]
list_var2 = [1, 2, 3]


trainset = LSTMMD.csvStockDataset(args.data_site, args.x_frames, args.y_frames, '2000-01-01', '2012-12-31')
valset = LSTMMD.csvStockDataset(args.data_site, args.x_frames, args.y_frames, '2013-01-01', '2016-12-31')
testset = LSTMMD.csvStockDataset(args.data_site, args.x_frames, args.y_frames, '2017-01-01', '2020-12-31')
partition = {'train': trainset, 'val': valset, 'test': testset}

setattr(args, name_var1, 0.0001)
setattr(args, name_var2, 1)
print(args)
setting, result = experiment(partition, deepcopy(args))




iteration=1
data_list = ['^KS11', '^KQ11','^IXIC','^GSPC','^DJI','^HSI',
             '^N225','^GDAXI','^FCHI','^IBEX','^TWII','^AEX',
             '^BSESN','^BVSP','GC=F','BTC-USD','ETH-USD']
save_file_path = 'C:\\Users\\USER\\PycharmProjects\\LSTM\\results'
for j in data_list:
    setattr(args, 'symbol', j)
    args.new_file_path = save_file_path +'\\'+'EncoderDecoderLSTM'+'_'+ args.symbol
    os.makedirs(args.new_file_path)
    if args.symbol == '^KQ11':
        train_start = (2013, 3, 3)
        train_end = (2016, 12, 31)
        val_start = (2017, 1, 1)
        val_end = (2018, 12, 31)
        test_start = (2019, 1, 1)
        test_end = (2020, 12, 31)
    elif args.symbol == 'CL=F':
        train_start = (2000, 8, 23)
        train_end = (2012, 12, 31)
        val_start = (2013, 1, 1)
        val_end = (2016, 12, 31)
        test_start = (2017, 1, 1)
        test_end = (2020, 12, 31)
    elif args.symbol == 'BTC-USD':
        train_start = (2014, 9, 17)
        train_end = (2018, 12, 31)
        val_start = (2019, 1, 1)
        val_end = (2019, 12, 31)
        test_start = (2020, 1, 1)
        test_end = (2020, 12, 31)
    elif args.symbol == 'ETH-USD':
        train_start = (2015, 8, 7)
        train_end = (2018, 12, 31)
        val_start = (2019, 1, 1)
        val_end = (2019, 12, 31)
        test_start = (2020, 1, 1)
        test_end = (2020, 12, 31)
    else:
        train_start = (2000, 1, 1)
        train_end = (2012, 12, 31)
        val_start = (2013, 1, 1)
        val_end = (2016, 12, 31)
        test_start = (2017, 1, 1)
        test_end = (2020, 12, 31)

    ## 데이터 load.
    trainset = LSTMMD.StockDataset(args.symbol, args.x_frames, args.y_frames, train_start, train_end)
    valset = LSTMMD.StockDataset(args.symbol, args.x_frames, args.y_frames, val_start, val_end)
    testset = LSTMMD.StockDataset(args.symbol, args.x_frames, args.y_frames, test_start, test_end)
    partition = {'train': trainset, 'val': valset, 'test': testset}

    test_value_list = []
    for k, iteration_n in enumerate(range(iteration)):
        args.iteration = iteration_n
        args.innate_path = args.new_file_path + '\\' + str(args.iteration) +'_iter' ## 내부 파일경로
        os.makedirs(args.innate_path)
        print(args)
        ## 실험을 실행했다.
        setting, result = experiment(partition, deepcopy(args))
        test_value_list.append(result['test_loss'])

        fig = plt.figure()
        plt.plot(result['train_losses'])
        plt.plot(result['val_losses'])
        plt.legend(['train_losses', 'val_losses'], fontsize=15)
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('loss', fontsize=15)
        plt.grid()
        plt.savefig(args.new_file_path + '\\' + str(args.iteration) + '_fig' + '.png')
        plt.close(fig)

        ## 이경우에
        ## 지정한 모델과, 지정한 데이터셋에 대한 결과가 저장되게된다.
        ## 지금 하려고 하는것은 result directory에 modek 마다, dataset 마다 iteration  원하는 데이터셋에
        #save_exp_result(setting, result)
    avg_test_value = sum(test_value_list)/len(test_value_list)
    std_test_value = np.std(test_value_list)
    with open(args.new_file_path + '\\' + 'result_t.txt', 'w') as f:
        print('avg: {}, std :{}'.format(avg_test_value, std_test_value), file=f)
    print('{}_{} 30 avg_test_value_list : {}'.format('EncoderDecoderLSTM',args.symbol ,avg_test_value))




df = load_exp_result('exp1_lr-ac')
df.columns


parameters = {'xtick.labelsize': 15,
          'ytick.labelsize': 15}
plt.rcParams.update(parameters)

plt.plot(df[["train_losses"]].values[0][0])
plt.plot(df[["val_losses"]].values[0][0])
plt.legend(['train_losses', 'val_losses'],fontsize=15)
plt.xlabel('epoch',fontsize=15)
plt.ylabel('loss',fontsize=15)
plt.grid()


import os
os.getcwd()
os.chdir('./LSTM')
os.chdir('C:\\Users\\lee\\PycharmProjects')



data = LSTMMD.StockDataset('GC=F', 10, 1, (2000, 1, 1), (2020, 12, 31))

data_list = []
for i in range(len(data)):
    data_list.append(data[i][1][0][0])

df = pd.DataFrame(data_list)

parameters = {'xtick.labelsize': 15,
          'ytick.labelsize': 15}
plt.rcParams.update(parameters)

plt.plot(df)




from scipy.stats import skew, kurtosis
## statistics

def statistics(data):
    mean = sum(data)/len(data)
    sd = np.std(data)
    ske = skew(data)
    kur = kurtosis(data, fisher=False)
    print('mean:{0:3.3f}, sd:{1:3.3f}, skew:{2:3.3f},kurto :{3:3.3f}'.format(mean,sd,ske,kur))


f = open('C:\\Users\\lee\\PycharmProjects\\LSTM\\data\\us_10y_tb.csv', 'r', encoding='cp949')
f = open('C:\\Users\\lee\\PycharmProjects\\LSTM\\data\\us_10y_tb.csv', 'r', encoding='cp949')
rdr = csv.reader(f)
data = []
for line in rdr:
    # print(line)
    data.append(line)
f.close()
data = pd.DataFrame(data)
print(data)
data_columns = list(data.iloc[0, :])
data = data.iloc[1:, :]
data_date = list(data.iloc[:, 0])
data_time_ind = pd.DatetimeIndex(data_date)
data = data.iloc[:, 1:7]
df = pd.DataFrame(np.array(data), index=data_time_ind, columns=data_columns[1:])

data = df.astype({'종가': 'float'})
data = data.loc[:, '종가']
statistics(data)
df


from statsmodels.tsa.stattools import adfuller
result = adfuller(data)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


from statsmodels.stats.stattools import jarque_bera
print(jarque_bera(data))