import sys
import pandas as pd
sys.path.append('C:\\Users\\leete\\PycharmProjects\\LSTM')
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import argparse
from copy import deepcopy # Add Deepcopy for args
import matplotlib.pyplot as plt
import LSTM_MODEL_DATASET as LSTMMD
from LSTM_MODEL_DATASET import metric1 as metric1
from LSTM_MODEL_DATASET import metric2 as metric2
from LSTM_MODEL_DATASET import metric3 as metric3
from LSTM_MODEL_DATASET import StockDatasetCV as StockDatasetCV
from LSTM_MODEL_DATASET import CV_Data_Spliter as CV_Data_Spliter
from LSTM_MODEL_DATASET import CV_train_Spliter as CV_train_Spliter
import os
import csv


class LSTM_enc(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_size):
        super(LSTM_enc, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        ## 파이토치에 있는 lstm모듈
        ## output dim 은 self.regressor에서 사용됨
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))


    def forward(self, x):
        # 새로 opdate 된 self.hidden과 lstm_out을 return 해줌
        # self.hidden 각각의 layer의 모든 hidden state 를 갖고있음
        ## LSTM의 hidden state에는 tuple로 cell state포함, 0번째는 hidden state tensor, 1번째는 cell state
        lstm_out, self.hidden = self.lstm(x, self.hidden)

        return lstm_out, self.hidden




class LSTM_dec(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_size, dropout, use_bn):
        super(LSTM_dec, self).__init__()
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
        self.regressor = self.make_regressor()

    def make_regressor(self): # 간단한 MLP를 만드는 함수
        layers = []
        if self.use_bn:
            layers.append(nn.BatchNorm1d(self.hidden_dim))  ##  nn.BatchNorm1d
        layers.append(nn.Dropout(self.dropout))    ##  nn.Dropout

        ## hidden dim을 outputdim으로 바꿔주는 MLP
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        regressor = nn.Sequential(*layers)
        return regressor

    def forward(self, x, encoder_hidden_states):
        # 새로 opdate 된 self.hidden과 lstm_out을 return 해줌
        # self.hidden 각각의 layer의 모든 hidden state 를 갖고있음
        ## LSTM의 hidden state에는 tuple로 cell state포함, 0번째는 hidden state tensor, 1번째는 cell state

        lstm_out, self.hidden = self.lstm(x, encoder_hidden_states)
        ## lstm_out : 각 time step에서의 lstm 모델의 output 값
        ## lstm_out[-1] : 맨마지막의 아웃풋 값으로 그 다음을 예측하고싶은 것이기 때문에 -1을 해줌
        y_pred = self.regressor(lstm_out[-1].view(self.batch_size, -1)) ## self.batch size로 reshape해 regressor에 대입

        return y_pred


def train(encoder,decoder, partition, enc_optimizer,dec_optimizer, loss_fn, args):
    trainloader = DataLoader(partition['train'],    ## DataLoader는 dataset에서 불러온 값으로 랜덤으로 배치를 만들어줌
                             batch_size=args.batch_size,
                             shuffle=False, drop_last=True)
    encoder.train()
    decoder.train()
    encoder.zero_grad()
    decoder.zero_grad()
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()


    not_used_data_len = len(partition['train']) % args.batch_size
    train_loss = 0.0
    y_pred_graph = []
    for i, (X, y, min, max) in enumerate(trainloader):

        ## (batch size, sequence length, input dim)
        ## x = (10, n, 6) >> x는 n일간의 input
        ## y= (10, m, 1) or (10, m)  >> y는 m일간의 종가를 동시에 예측
        ## lstm은 한 스텝별로 forward로 진행을 함
        ## (sequence length, batch size, input dim) >> 파이토치 default lstm은 첫번째 인자를 sequence length로 받음
        ## x : [n, 10, 6], y : [m, 10]

        X = X.transpose(0, 1).unsqueeze(-1).float().to(args.device)
        ## transpose는 seq length가 먼저 나와야 하기 때문에 0번째와 1번째를 swaping
        encoder.zero_grad()
        enc_optimizer.zero_grad()
        encoder.hidden = [hidden.to(args.device) for hidden in encoder.init_hidden()]
        y_pred_enc, encoder_hidden = encoder(X)

        decoder_hidden = encoder_hidden


        y_true = y[:, :].float().to(args.device)  ## index-3은 종가를 의미(dataframe 상에서)
        #print(torch.max(X[:, :, 3]), torch.max(y_true))


        decoder.zero_grad()
        dec_optimizer.zero_grad()

        y_pred_dec = decoder(X, decoder_hidden)
        reformed_y_pred = y_pred_dec.squeeze() * (max - min) + min

        y_pred_graph = y_pred_graph + reformed_y_pred.tolist()
        loss = loss_fn(y_pred_dec.view(-1), y_true.view(-1)) # .view(-1)은 1열로 줄세운것
        loss.backward()  ## gradient 계산
        enc_optimizer.step()  ## parameter 갱신
        dec_optimizer.step() ## parameter를 update 해줌 (.backward() 연산이 시행된다면(기울기 계산단계가 지나가면))

        train_loss += loss.item()   ## item()은 loss의 스칼라값을 칭하기때문에 cpu로 다시 넘겨줄 필요가 없다.

    train_loss = train_loss / len(trainloader)
    return encoder, decoder, train_loss, y_pred_graph, not_used_data_len


def validate(encoder, decoder, partition, loss_fn, args):
    valloader = DataLoader(partition['val'],
                           batch_size=args.batch_size,
                           shuffle=False, drop_last=True)

    not_used_data_len = len(partition['val']) % args.batch_size

    encoder.eval()
    decoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        y_pred_graph = []
        for i, (X, y, min, max) in enumerate(valloader):

            X = X.transpose(0, 1).unsqueeze(-1).float().to(args.device)
            encoder.hidden = [hidden.to(args.device) for hidden in encoder.init_hidden()]
            y_pred_enc, encoder_hidden = encoder(X)

            decoder_hidden = encoder_hidden

            y_true = y[:, :].float().to(args.device)

            y_pred_dec = decoder(X,decoder_hidden)

            reformed_y_pred = y_pred_dec.squeeze() * (max - min) + min
            y_pred_graph = y_pred_graph + reformed_y_pred.tolist()

            # print('validate y_pred: {}, y_pred.shape : {}'. format(y_pred, y_pred.shape))
            loss = loss_fn(y_pred_dec.view(-1), y_true.view(-1))

            val_loss += loss.item()

    val_loss = val_loss / len(valloader) ## 한 배치마다의 로스의 평균을 냄
    return val_loss, y_pred_graph, not_used_data_len   ## 그결과값이 한 에폭마다의 LOSS

def test(encoder,decoder, partition, args):
    testloader = DataLoader(partition['test'],
                           batch_size=args.batch_size,
                           shuffle=False, drop_last=True)
    not_used_data_len = len(partition['test']) % args.batch_size

    encoder.eval()
    decoder.eval()

    test_loss_metric1 = 0.0
    test_loss_metric2 = 0.0
    test_loss_metric3 = 0.0
    with torch.no_grad():
        y_pred_graph = []
        for i, (X, y, min, max) in enumerate(testloader):

            X = X.transpose(0, 1).unsqueeze(-1).float().to(args.device)
            encoder.hidden = [hidden.to(args.device) for hidden in encoder.init_hidden()]
            y_pred_enc, encoder_hidden = encoder(X)

            decoder_hidden = encoder_hidden

            y_true = y[:, :].float().to(args.device)

            y_pred_dec = decoder(X,decoder_hidden)

            reformed_y_pred = y_pred_dec.squeeze() * (max - min) + min
            reformed_y_true = y_true.squeeze() * (max - min) + min
            y_pred_graph = y_pred_graph + reformed_y_pred.tolist()


            test_loss_metric1 += metric1(reformed_y_pred, reformed_y_true)
            test_loss_metric2 += metric2(reformed_y_pred, reformed_y_true)
            test_loss_metric3 += metric3(reformed_y_pred, reformed_y_true)

    test_loss_metric1 = test_loss_metric1 / len(testloader)
    test_loss_metric2 = test_loss_metric2 / len(testloader)
    test_loss_metric3 = test_loss_metric3 / len(testloader)
    return test_loss_metric1, test_loss_metric2, test_loss_metric3, y_pred_graph, not_used_data_len


def experiment(partition, args):
    encoder = args.encoder(args.input_dim, args.hid_dim, args.n_layers, args.batch_size)
    decoder = args.decoder(args.input_dim, args.hid_dim, args.y_frames, args.n_layers, args.batch_size,
                           args.dropout, args.use_bn)

    encoder.to(args.device)
    decoder.to(args.device)


    loss_fn = nn.MSELoss()
    # loss_fn.to(args.device) ## gpu로 보내줌  간혹 loss에 따라 안되는 경우도 있음
    if args.optim == 'SGD':
        enc_optimizer = optim.SGD(encoder.parameters(), lr=args.lr, weight_decay=args.l2)
        dec_optimizer = optim.SGD(decoder.parameters(), lr=args.lr, weight_decay=args.l2)
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
    ## 우리는 지금 epoch 마다 모델을 저장해야 하기때문에 여기에 저장하는 기능을 넣어야함.
    ## 실제로 우리는 디렉토리를 만들어야함
    ## 모델마다의 디렉토리를 만들어야하는데
    epoch_graph_list = []
    for epoch in range(args.epoch):  # loop over the dataset multiple times
        ts = time.time()
        encoder, decoder, train_loss, graph1, unused_triain = train(encoder, decoder, partition,
                                                                    enc_optimizer, dec_optimizer, loss_fn, args)
        val_loss, graph2, unused_val = validate(encoder, decoder, partition, loss_fn, args)
        te = time.time()

        epoch_graph_list.append([graph1,graph2])
        # ====== Add Epoch Data ====== # ## 나중에 그림그리는데 사용할것
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # ============================ #
        ## 각 에폭마다 모델을 저장하기 위한 코드
        torch.save(encoder.state_dict(), args.innate_path + '\\' + str(epoch) +'_epoch' +'_encoder' +'.pt')
        torch.save(decoder.state_dict(), args.innate_path + '\\' + str(epoch) +'_epoch' +'_decoder' +'.pt')

        print('Epoch {}, Loss(train/val) {:2.5f}/{:2.5f}. Took {:2.2f} sec, Iteration {}'
              .format(epoch, train_loss, val_loss, te - ts, args.iteration))

    ## 여기서 구하는것은 val_losses에서 가장 값이 최소인 위치를 저장함
    site_val_losses = val_losses.index(min(val_losses)) ## 10 epoch일 경우 0번째~9번째 까지로 나옴
    encoder = args.encoder(args.input_dim, args.hid_dim, args.n_layers, args.batch_size)
    decoder = args.decoder(args.input_dim, args.hid_dim, args.y_frames, args.n_layers, args.batch_size, args.dropout,
                           args.use_bn)

    encoder.to(args.device)
    decoder.to(args.device)

    encoder.load_state_dict(torch.load(args.innate_path + '\\' + str(site_val_losses) +'_epoch'+'_encoder' + '.pt'))
    decoder.load_state_dict(torch.load(args.innate_path + '\\' + str(site_val_losses) +'_epoch' +'_decoder'+ '.pt'))

    ## graph
    train_val_graph = epoch_graph_list[site_val_losses]

    test_loss_metric1, test_loss_metric2, test_loss_metric3, graph3, unused_test = test(encoder, decoder, partition, args)
    print('test_loss_metric1: {},\n test_loss_metric2: {}, \ntest_loss_metric3: {}'
          .format(test_loss_metric1, test_loss_metric2, test_loss_metric3))

    with open(args.innate_path + '\\'+ str(site_val_losses)+'Epoch_test_metric' +'.csv', 'w') as fd:
        print('test_loss_metric1 : {} \n test_loss_metric2 : {} \n test_loss_metric3 : {}'
              .format(test_loss_metric1, test_loss_metric2, test_loss_metric3), file=fd)
    # ======= Add Result to Dictionary ======= #
    result = {}
    result['train_losses'] = train_losses
    result['val_losses'] = val_losses
    result['test_loss_metric1'] = test_loss_metric1
    result['test_loss_metric2'] = test_loss_metric2
    result['test_loss_metric3'] = test_loss_metric3
    result['train_val_graph'] = train_val_graph
    result['test_graph'] = graph3
    result['unused_data'] = [unused_triain,unused_val,unused_test]

    return vars(args), result      ## vars(args) 1: args에있는 attrubute들을 dictionary 형태로 보길 원한다면 vars 함


# ====== Random Seed Initialization ====== #
seed = 666
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args("")  ## ""을 써주는 이유는 터미널이 아니기때문에
args.exp_name = "exp1_lr"
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== Data Loading ====== #
args.symbol = '^KQ11'
args.batch_size = 128
args.x_frames = 20
args.y_frames = 1
args.encoder = LSTM_enc
args.decoder = LSTM_dec

# ====== Model Capacity ===== #
args.input_dim = 1
args.hid_dim = 10
args.n_layers = 1

# ====== Regularization ======= #
args.l2 = 0.00001
args.dropout = 0.0
args.use_bn = True

# ====== Optimizer & Training ====== #
args.optim = 'Adam'  # 'RMSprop' #SGD, RMSprop, ADAM...
args.lr = 0.0001
args.epoch = 2
args.split = 4
# ====== Experiment Variable ====== #
## csv 파일 실행
#trainset = LSTMMD.csvStockDataset(args.data_site, args.x_frames, args.y_frames, '2000-01-01', '2012-12-31')
#valset = LSTMMD.csvStockDataset(args.data_site, args.x_frames, args.y_frames, '2013-01-01', '2016-12-31')
#testset = LSTMMD.csvStockDataset(args.data_site, args.x_frames, args.y_frames, '2017-01-01', '2020-12-31')
#partition = {'train': trainset, 'val': valset, 'test': testset}

# '^KS11' : KOSPI
# '^KQ11' : 코스닥
# '^IXIC' : 나스닥
# '^GSPC' : SNP 500 지수
# '^DJI' : 다우존수 산업지수
# '^HSI' : 홍콩 항생 지수
# '^N225' : 니케이지수
# '^GDAXI' : 독일 DAX
# '^FTSE' : 영국 FTSE
# '^FCHI' : 프랑스 CAC
# '^IBEX' : 스페인 IBEX
# '^TWII' : 대만 기권
# '^AEX' : 네덜란드 AEX
# '^BSESN' : 인도 센섹스
# 'RTSI.ME' : 러시아 RTXI
# '^BVSP' : 브라질 보베스파 지수
# 'GC=F' : 금 가격
# 'CL=F' : 원유 가격 (2000/ 8 / 20일 부터 데이터가 있음)
# 10년만기 미국 국채
# 'BTC-USD' : 비트코인 암호화폐
# 'ETH-USD' : 이더리움 암호화폐




# model_list = [LSTMMD.RNN,LSTMMD.LSTM,LSTMMD.GRU]
# data_list = ['^KS11', '^KQ11','^IXIC','^GSPC','^DJI','^HSI',
#              '^N225','^GDAXI','^FCHI','^IBEX','^TWII','^AEX',
#              '^BSESN','^BVSP','GC=F','BTC-USD','ETH-USD']

data_list = ['ETH-USD','^KS11']
data_list = ['ETH-USD']

args.save_file_path = 'C:\\Users\\leete\\PycharmProjects\\LSTM\\results'

with open(args.save_file_path + '\\' + 'ENC_DEC_result_t.csv', 'w', encoding='utf-8', newline='') as f:
    wr = csv.writer(f)

    wr.writerow(["model", "stock", "avg_test_metric1", "std_test_metric1",
                                                       "avg_test_metric2", "std_test_metric2",
                                                       "avg_test_metric3", "std_test_metric3"])

    for j in data_list:
        setattr(args, 'symbol', j)
        model_name = "ENC_DEC_LSTM"
        args.new_file_path = args.save_file_path +'\\'+ model_name+'_' + args.symbol
        os.makedirs(args.new_file_path)
        if args.symbol == '^KQ11':
            data_start = (2013, 3, 3)
            data_end = (2020, 12, 31)
        elif args.symbol == 'CL=F':
            data_start = (2011, 1, 1)               ##(2000, 8, 23)
            data_end = (2020, 12, 31)
        elif args.symbol == 'BTC-USD':
            data_start = (2014, 9, 17)
            data_end = (2020, 12, 31)
        elif args.symbol == 'ETH-USD':
            data_start = (2015, 8, 7)
            data_end = (2020, 12, 31)
        else:  ## 나머지 모든 데이터들
            data_start = (2011, 1, 1)
            data_end = (2020, 12, 31)

        splitted_test_train = CV_Data_Spliter(args.symbol, data_start, data_end, n_splits=args.split)
        entire_data = splitted_test_train.entire_data()

        args.series_Data = splitted_test_train.entire_data
        test_metric1_list = []
        test_metric2_list = []
        test_metric3_list = []
        for iteration_n in range(args.split):
            args.iteration = iteration_n
            train_data, test_data = splitted_test_train[args.iteration][0], splitted_test_train[args.iteration][1]
            test_size = splitted_test_train.test_size
            splitted_train_val = CV_train_Spliter(train_data, args.symbol, test_size=test_size)
            train_data, val_data = splitted_train_val[1][0], splitted_train_val[1][1]

            trainset = StockDatasetCV(train_data, args.x_frames, args.y_frames)
            valset   = StockDatasetCV(val_data, args.x_frames, args.y_frames)
            testset  = StockDatasetCV(test_data, args.x_frames, args.y_frames)
            partition = {'train': trainset, 'val': valset, 'test': testset}

            args.innate_path = args.new_file_path + '\\' + str(args.iteration) +'_iter' ## 내부 파일경로
            os.makedirs(args.innate_path)
            print(args)

            setting, result = experiment(partition, deepcopy(args))
            test_metric1_list.append(result['test_loss_metric1'])
            test_metric2_list.append(result['test_loss_metric2'])
            test_metric3_list.append(result['test_loss_metric3'])

            ## 그림
            fig = plt.figure()
            plt.plot(result['train_losses'])
            plt.plot(result['val_losses'])
            plt.legend(['train_losses', 'val_losses'], fontsize=15)
            plt.xlabel('epoch', fontsize=15)
            plt.ylabel('loss', fontsize=15)
            plt.grid()
            plt.savefig(args.new_file_path + '\\' + str(args.iteration) + '_fig' + '.png')
            plt.close(fig)

            predicted_traing = result['train_val_graph'][0]
            predicted_valg = result['train_val_graph'][1]
            predicted_testg = result['test_graph']
            entire_dataa = entire_data['Close'].values.tolist()

            train_length = len(predicted_traing)
            val_length = len(predicted_valg)
            test_length = len(predicted_testg)
            entire_length = len(entire_dataa)

            unused_triain = result['unused_data'][0]
            unused_val = result['unused_data'][1]
            unused_test = result['unused_data'][2]

            train_index = list(range(args.x_frames,args.x_frames+train_length))
            val_index = list(range(args.x_frames+train_length+unused_triain+args.x_frames, args.x_frames+train_length+unused_triain+args.x_frames+val_length))
            test_index = list(range(args.x_frames+train_length+unused_triain+args.x_frames+val_length+unused_val+args.x_frames, args.x_frames+train_length+unused_triain+args.x_frames+val_length+unused_val+args.x_frames+test_length))
            entire_index = list(range(entire_length))


            fig2 = plt.figure()
            plt.plot(entire_index, entire_dataa)
            plt.plot(train_index, predicted_traing)
            plt.plot(val_index, predicted_valg)
            plt.plot(test_index, predicted_testg)
            plt.legend(['raw_data', 'predicted_train', 'predicted_val','predicted_test'], fontsize=15)
            plt.xlim(0, entire_length)
            plt.xlabel('time', fontsize=15)
            plt.ylabel('value', fontsize=15)
            plt.grid()
            plt.savefig(args.new_file_path + '\\' + str(args.iteration) + '_chart_fig' + '.png')
            plt.close(fig2)


            #save_exp_result(setting, result)

        avg_test_metric1 = sum(test_metric1_list) / len(test_metric1_list)
        avg_test_metric2 = sum(test_metric2_list) / len(test_metric2_list)
        avg_test_metric3 = sum(test_metric3_list) / len(test_metric3_list)
        std_test_metric1 = np.std(test_metric1_list)
        std_test_metric2 = np.std(test_metric2_list)
        std_test_metric3 = np.std(test_metric3_list)

        #csv파일에 기록하기
        wr.writerow([model_name, args.symbol, avg_test_metric1, std_test_metric1,
                                                   avg_test_metric2, std_test_metric2,
                                                   avg_test_metric3, std_test_metric3])

        with open(args.new_file_path + '\\' + 'result_t.txt', 'w') as fd:
            print('metric1 \n avg: {}, std : {}\n'.format(avg_test_metric1, std_test_metric1), file=fd)
            print('metric2 \n avg: {}, std : {}\n'.format(avg_test_metric2, std_test_metric2), file=fd)
            print('metric3 \n avg: {}, std : {}\n'.format(avg_test_metric3, std_test_metric3), file=fd)
        print('{}_{} 30 avg_test_value_list : {}'.format(model_name, args.symbol, avg_test_metric1))
        print('{}_{} 30 avg_test_value_list : {}'.format(model_name, args.symbol, avg_test_metric2))
        print('{}_{} 30 avg_test_value_list : {}'.format(model_name, args.symbol, avg_test_metric3))