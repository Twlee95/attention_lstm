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

import seaborn as sns
import matplotlib.pyplot as plt


print(torch.__version__)
pylab.rcParams['figure.figsize'] = (15, 9)


# We will look at stock prices over the past year, starting at January 1, 2016
start = (2000, 12, 1)
##시간을 그냥 튜플이 아닌 datetime 오브젝트로 넣어줘야함 ,
## 날짜관련 모듈
## *start  하나씩 하나씩을 순서대로 인자로 넣어주겠다
start = datetime.datetime(*start)
end = datetime.date.today()


# we can read data in data 크롤링, but we want to save time.
# therefore we use pdr.DataReader


nikei = pdr.DataReader('^N225', 'yahoo', start, end)
## what is output of the pdr.DataReader
print(nikei)

nikei.Low.plot(grid=True)
nikei.Close.plot(grid=True)




## 6차원짜리 feature

samsung.tail()
print(samsung.isna().sum())




## t+1 부터 t+5 까지 동시에 예측
## data preperation

## dataset은 i번째 record 값을 출력해주는 역할을 함.
## if input feature is 2 demension, then 인풋으로는 2차원을 주고, 아웃풋으로는 1차원 값 하나를 return 하면 끝나는 역할

## batch를 만들어주는것이 pytorch의 data loader임, random sample도 만들어줌
## data loader는 파이토치에 구현되어있음
class StockDataset(Dataset):

    def __init__(self, symbol, x_frames, y_frames, start, end):
        self.symbol = symbol
        self.x_frames = x_frames
        self.y_frames = y_frames

        self.start = datetime.datetime(*start)
        self.end = datetime.datetime(*end)

        self.data = pdr.DataReader(self.symbol, 'yahoo', self.start, self.end)
        print(self.data.isna().sum())
    ## 데이터셋에 len() 을 사용하기 위해 만들어주는것 (dataloader에서 batch를 만들때 이용됨)
    def __len__(self):
        return len(self.data) - (self.x_frames + self.y_frames) + 1

    ## a[:]와 같은 indexing 을 위해 getinem 을 만듬
    ## custom dataset이 list가 아님에도 그 데이터셋의 i번째의 x,y를 출력해줌
    def __getitem__(self, idx):
        idx += self.x_frames
        ## iloc
        data = self.data.iloc[idx - self.x_frames:idx + self.y_frames]
        data = data[['High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume']] ## 컬럼순서맞추기 위해 한것
        ## nomalization
        data = data.apply(lambda x: np.log(x + 1) - np.log(x[self.x_frames - 1] + 1))
        data = data.values ## (data.frame >> numpy array) convert >> 나중에 dataloader가 취합해줌
        ## x와 y 기준으로 split
        X = data[:self.x_frames]
        y = data[self.x_frames:]

        return X, y


# X : (2, 10, 6)
# (batch size, 최근 10일, feature size)
# y : (2, 5, 6)
# (batch size, 이후 5일, feature size)



## model define
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_size, dropout, use_bn):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.batch_size = batch_size
        self.dropout = dropout
        self.use_bn = use_bn

        ## 파이토치에 있는 lstm모듈
        ## output dim 은 self.regressor에서 사용됨
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.hidden = self.init_hidden()
        self.regressor = self.make_regressor()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def make_regressor(self): # 간단한 MLP를 만드는 함수
        layers = []
        if self.use_bn:
            layers.append(nn.BatchNorm1d(self.hidden_dim))  ##  nn.BatchNorm1d
        layers.append(nn.Dropout(self.dropout))    ##  nn.Dropout
        ## hidden dim을 outputdim으로 바꿔주는 MLP
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim // 2, self.output_dim)) # 여기서 output dim이 사용됨
        regressor = nn.Sequential(*layers)

        return regressor

    def forward(self, x):
        # 새로 opdate 된 self.hidden과 lstm_out을 return 해줌
        # self.hidden 각각의 layer의 모든 hidden state 를 갖고있음

        ## LSTM의 hidden state에는 tuple로 cell state포함, 0번째는 hidden state tensor, 1번째는 cell state
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        print('lstm_out: {}, lstm_out shape{}'.format(lstm_out,lstm_out.shape))
        print('self.hidden : {}'.format(self.hidden))
        # lstm_out : 각 time step에서의 lstm 모델의 output 값
        ## lstm_out[-1] : 맨마지막의 아웃풋 값으로 그 다음을 예측하고싶은 것이기 때문에 -1을 해줌
        y_pred = self.regressor(lstm_out[-1].view(self.batch_size, -1)) ## self.batch size로 reshape해 regressor에 대입
        print('lstm_out[-1]: {}, shape : {}'.format(lstm_out[-1], lstm_out[-1].shape))
        print('lstm_out[-1].view(self.batch_size, -1): {}, shape : {}'.format(lstm_out[-1].view(self.batch_size, -1), lstm_out[-1].view(self.batch_size, -1).shape))
        print('lstm y_pred: {},lstm y_pred shape : {}'.format(y_pred, y_pred.shape))
        return y_pred



def metric(y_pred, y_true):
    perc_y_pred = np.exp(y_pred.cpu().detach().numpy())
    perc_y_true = np.exp(y_true.cpu().detach().numpy())
    # mean_absolute_error : 차이의 절댓값을 loss function으로 사용
    mae = mean_absolute_error(perc_y_true, perc_y_pred, multioutput='raw_values')
    return (1-mae) * 100


def train(model, partition, optimizer, loss_fn, args):
    trainloader = DataLoader(partition['train'],    ## DataLoader는 dataset에서 불러온 값으로 랜덤으로 배치를 만들어줌
                             batch_size=args.batch_size,
                             shuffle=True, drop_last=True)
    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    train_acc = 0.0
    train_loss = 0.0
    for i, (X, y) in enumerate(trainloader):

        ## (batch size, sequence length, input dim)
        ## x = (10, n, 6) >> x는 n일간의 input
        ## y= (10, m, 1) or (10, m)  >> y는 m일간의 종가를 동시에 예측
        ## lstm은 한 스텝별로 forward로 진행을 함
        ## (sequence length, batch size, input dim) >> 파이토치 default lstm은 첫번째 인자를 sequence length로 받음
        ## x : [n, 10, 6], y : [m, 10]

        X = X.transpose(0, 1).float().to(args.device) ## transpose는 seq length가 먼저 나와야 하기 때문에 0번째와 1번째를 swaping
        y_true = y[:, :, 3].float().to(args.device)  ## index-3은 종가를 의미(dataframe 상에서)

        model.zero_grad()
        optimizer.zero_grad()
        model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

        y_pred = model(X)
        loss = loss_fn(y_pred.view(-1), y_true.view(-1)) # .view(-1)은 1열로 줄세운것
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # print('train_loss : {}'.format(train_loss))
        # print('metric(y_pred, y_true) : {}'.format(metric(y_pred, y_true)))
        train_acc += metric(y_pred, y_true)[0]
        # print('train_acc : {}'.format(train_acc))

    train_loss = train_loss / len(trainloader)
    train_acc = train_acc / len(trainloader)
    return model, train_loss, train_acc


def validate(model, partition, loss_fn, args):
    valloader = DataLoader(partition['val'],
                           batch_size=args.batch_size,
                           shuffle=False, drop_last=True)
    model.eval()

    val_acc = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for i, (X, y) in enumerate(valloader):

            X = X.transpose(0, 1).float().to(args.device)
            y_true = y[:, :, 3].float().to(args.device)
            model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

            y_pred = model(X)
            loss = loss_fn(y_pred.view(-1), y_true.view(-1))

            val_loss += loss.item()
            val_acc += metric(y_pred, y_true)[0]

    val_loss = val_loss / len(valloader)
    val_acc = val_acc / len(valloader)
    return val_loss, val_acc

def test(model, partition, args):
    testloader = DataLoader(partition['test'],
                           batch_size=args.batch_size,
                           shuffle=False, drop_last=True)
    model.eval()

    test_acc = 0.0
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):

            X = X.transpose(0, 1).float().to(args.device)
            y_true = y[:, :, 3].float().to(args.device)
            model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

            y_pred = model(X)
            print('y_pred: {},shape : {}'.format(y_pred,y_pred.shape))
            print('metric(y_pred, y_true): {},shape : {}'.format(metric(y_pred, y_true), metric(y_pred, y_true).shape))
            print('metric(y_pred, y_true)[0]: {},shape : {}'.format(metric(y_pred, y_true)[0], metric(y_pred, y_true)[0].shape))
            test_acc += metric(y_pred, y_true)[0]

    test_acc = test_acc / len(testloader)
    return test_acc


def experiment(partition, args):
    model = LSTM(args.input_dim, args.hid_dim, args.y_frames, args.n_layers, args.batch_size, args.dropout, args.use_bn)
    model.to(args.device)
    loss_fn = torch.nn.MSELoss()

    loss_fn = nn.MSELoss()
    if args.optim == 'SGD':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise ValueError('In-valid optimizer choice')

    # ===== List for epoch-wise data ====== #
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    # ===================================== #

    for epoch in range(args.epoch):  # loop over the dataset multiple times
        ts = time.time()
        model, train_loss, train_acc = train(model, partition, optimizer, loss_fn, args)
        val_loss, val_acc = validate(model, partition, loss_fn, args)
        te = time.time()

        # ====== Add Epoch Data ====== #
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        # ============================ #

        print(
            'Epoch {}, Acc(train/val): {:2.2f}/{:2.2f}, Loss(train/val) {:2.5f}/{:2.5f}. Took {:2.2f} sec'.format(epoch,
                                                                                                                  train_acc,
                                                                                                                  val_acc,
                                                                                                                  train_loss,
                                                                                                                  val_loss,
                                                                                                                  te - ts))

    test_acc = test(model, partition, args)

    # ======= Add Result to Dictionary ======= #
    result = {}
    result['train_losses'] = train_losses
    result['val_losses'] = val_losses
    result['train_accs'] = train_accs
    result['val_accs'] = val_accs
    result['train_acc'] = train_acc
    result['val_acc'] = val_acc
    result['test_acc'] = test_acc
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
    dir_path = './results'
    filenames = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) if '.json' in f]
    list_result = []
    for filename in filenames:
        if exp_name in filename:
            with open(join(dir_path, filename), 'r') as infile:
                results = json.load(infile)
                list_result.append(results)
    df = pd.DataFrame(list_result)  # .drop(columns=[])
    return df


def plot_acc(var1, var2, df):
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(15, 6)
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    sns.barplot(x=var1, y='train_acc', hue=var2, data=df, ax=ax[0])
    sns.barplot(x=var1, y='val_acc', hue=var2, data=df, ax=ax[1])
    sns.barplot(x=var1, y='test_acc', hue=var2, data=df, ax=ax[2])

    ax[0].set_title('Train Accuracy')
    ax[1].set_title('Validation Accuracy')
    ax[2].set_title('Test Accuracy')


def plot_loss_variation(var1, var2, df, **kwargs):
    list_v1 = df[var1].unique()
    list_v2 = df[var2].unique()
    list_data = []

    for value1 in list_v1:
        for value2 in list_v2:
            row = df.loc[df[var1] == value1]
            row = row.loc[df[var2] == value2]

            train_losses = list(row.train_losses)[0]
            val_losses = list(row.val_losses)[0]

            for epoch, train_loss in enumerate(train_losses):
                list_data.append({'type': 'train', 'loss': train_loss, 'epoch': epoch, var1: value1, var2: value2})
            for epoch, val_loss in enumerate(val_losses):
                list_data.append({'type': 'val', 'loss': val_loss, 'epoch': epoch, var1: value1, var2: value2})

    df = pd.DataFrame(list_data)
    g = sns.FacetGrid(df, row=var2, col=var1, hue='type', **kwargs)
    g = g.map(plt.plot, 'epoch', 'loss', marker='.')
    g.add_legend()
    g.fig.suptitle('Train loss vs Val loss')
    plt.subplots_adjust(top=0.89)  # 만약 Title이 그래프랑 겹친다면 top 값을 조정해주면 됩니다! 함수 인자로 받으면 그래프마다 조절할 수 있겠죠?


def plot_acc_variation(var1, var2, df, **kwargs):
    list_v1 = df[var1].unique()
    list_v2 = df[var2].unique()
    list_data = []

    for value1 in list_v1:
        for value2 in list_v2:
            row = df.loc[df[var1] == value1]
            row = row.loc[df[var2] == value2]

            train_accs = list(row.train_accs)[0]
            val_accs = list(row.val_accs)[0]
            test_acc = list(row.test_acc)[0]

            for epoch, train_acc in enumerate(train_accs):
                list_data.append({'type': 'train', 'Acc': train_acc, 'test_acc': test_acc, 'epoch': epoch, var1: value1,
                                  var2: value2})
            for epoch, val_acc in enumerate(val_accs):
                list_data.append(
                    {'type': 'val', 'Acc': val_acc, 'test_acc': test_acc, 'epoch': epoch, var1: value1, var2: value2})

    df = pd.DataFrame(list_data)
    g = sns.FacetGrid(df, row=var2, col=var1, hue='type', **kwargs)
    g = g.map(plt.plot, 'epoch', 'Acc', marker='.')

    def show_acc(x, y, metric, **kwargs):
        plt.scatter(x, y, alpha=0.3, s=1)
        metric = "Test Acc: {:1.3f}".format(list(metric.values)[0])
        plt.text(0.05, 0.95, metric, horizontalalignment='left', verticalalignment='center',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='yellow', alpha=0.5, boxstyle="round,pad=0.1"))

    g = g.map(show_acc, 'epoch', 'Acc', 'test_acc')

    g.add_legend()
    g.fig.suptitle('Train Accuracy vs Val Accuracy')
    plt.subplots_adjust(top=0.89)





# ====== Random Seed Initialization ====== #
seed = 666
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args("")
args.exp_name = "exp1_lr"
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== Data Loading ====== #
args.symbol = '^HSI'
args.batch_size = 128
args.x_frames = 10
args.y_frames = 1

# ====== Model Capacity ===== #
args.input_dim = 6
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

trainset = StockDataset(args.symbol, args.x_frames, args.y_frames, (2000, 1, 1), (2012, 1, 1))
valset = StockDataset(args.symbol, args.x_frames, args.y_frames, (2012, 1, 1), (2016, 1, 1))
testset = StockDataset(args.symbol, args.x_frames, args.y_frames, (2016, 1, 1), (2019, 2, 1))
partition = {'train': trainset, 'val': valset, 'test': testset}


len(trainset)+len(valset)+len(testset)


for var1 in list_var1:
    for var2 in list_var2:
        setattr(args, name_var1, var1)
        setattr(args, name_var2, var2)
        print(args)

        setting, result = experiment(partition, deepcopy(args))
        save_exp_result(setting, result)


var1 = 'lr'
var2 = 'n_layers'
df = load_exp_result('exp1')
plot_acc(var1, var2, df)
plot_loss_variation(var1, var2, df, sharey=False) #sharey를 True로 하면 모둔 subplot의 y축의 스케일이 같아집니다.
plot_acc_variation(var1, var2, df, margin_titles=True, sharey=True)



import os
os.getcwd()
os.chdir('./LSTM')



from functools import reduce
primes = [2, 3, 5, 7, 11, 13]

def product(*numbers):
    p = reduce(lambda x, y: x * y, numbers)
    return p

product(*primes)
# 30030

product(primes)
# [2, 3, 5, 7, 11, 13]