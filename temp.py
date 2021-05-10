## 가상의 데이터셋
import numpy as np
from torch.utils.data import Dataset, DataLoader


# manage experiment
import hashlib
import json ## 파일로 저장하는 dictionary
from os import listdir
from os.path import isfile, join
import pandas as pd


def save_exp_result(setting, result):
    exp_name = setting['exp_name']
    del setting['epoch']   ## epoch이 바뀌어도 다른 파일이 생기지 않도록 하는 코드(hash를 만들 때 고려가 안되도록)

    hash_key = hashlib.sha1(str(setting).encode()).hexdigest()[:6]
    filename = './results/{}-{}.json'.format(exp_name, hash_key)
    result.update(setting)     ## result라는 dictionary에 실험결과와함께 실험 setting도 저장하고싶기때문에 dic+dic >>.update를 사용
    with open(filename, 'w') as f:    ## 이렇게하면 저장이 끝남 ('w'는 쓰기모드이다, 'r' : 읽기모드)
        json.dump(result, f)

## linux 같은 terminal 이었으면 폴더에 들어가서 text.json이 저장이 되어 있었어야함

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





## 실험결과 load
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

class dummyset(Dataset):
    def __init__(self, num_data):
        self.x = np.array(list(range(num_data*2))).reshape(-1, 2)
        self.y = np.array(list(range(num_data)))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


dataset= dummyset(100)
print(dataset.x)
print(dataset.y)
print(len(dataset))

print(dataset[0])

dataloader = DataLoader(dataset, 3, shuffle= True, drop_last=True)
## dataloader는 list로 되어있다.
## drop_last를 true로 놓는 이유는 마지막 관측치를 drop해야함 -lstm에서

for X, y in dataloader:
    print(X.shape, y.shape)


class StockDataset(Dataset):

    def __init__(self, x_frames, y_frames):
        self.x_frames = x_frames
        self.y_frames = y_frames

        self.data = NN255
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

data[['Close', 'Adj Close', 'Volume']]
data[['High', 'Low', 'Open']]



data = NN255.iloc[13-10:13+5]
data = data[['High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume']]
data = data.astype(float)

data = data.apply(lambda x: np.log(x + 1) - np.log(x[10 - 1] + 1))
data = data.values

X = data[:10]
y = data[10:]

samsung

data2 = samsung.iloc[13-10:13+5]
data2 = data[['High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume']]

data2 = data.apply(lambda x: np.log(x + 1) - np.log(x[10 - 1] + 1))
data2 = data.values

X = data[:10]
y = data[10:]



type(data)


type(data2)


dataset = StockDataset(10, 5)
print(dataset)
dataloader = DataLoader(dataset, 2, drop_last=True, shuffle= True)


dataloader[0]
print(dir(dataloader))
for X, y in dataloader:
    print(X.shape, y.shape)



## json
import json
a = {'vlaue': 5, 'value2':10, 'seq': [1,2,3,4,5]}
#a = {'vlaue': 5,  'seq': [1,2,3,4,5]}
filename = 'test.json'

## 쓰기                         ## with가 없으면 f.close()를 해줘야한다. ex) 파일이 열려있으므로
with open(filename, 'w') as f: ## with 없이 open을 쓰게되면 파이썬은 항상 파일을 열어놓은 상태로 두게 된다.
    jsonp = json.dump(a,f)             ##  dumps() 함수: Python 객체를 JSON 문자열로 변환
    print(jsonp)

with open(filename, 'r') as f: ## 항상 썼을때는 제대로 읽어졌는지 확인해야함
    result = json.load(f)      ## loads() 함수: JSON 문자열을 Python 객체로 변환
    print(result)


                ## hash란 임의 길이의 문자열이 들어왔을 때 고정된 길이의 문자열로 바꿔주는 함수
import hashlib  ## 이 문자열은 알파벳과 숫자로 구성됨 >> 같은 인풋에는 같은 아웃풋이 나옴
a = "my name is taewon"
hash_key = hashlib.sha1(a.encode()).hexdigest()[:6]    ##hexdigest() : 보기쉬운 문자열로 출력해줌// [] : 6개까지
        ##해쉬로 뿌숴버림(문자열을 encoding후)
        ## 인풋이 조금만 바껴도 결과과 완전히 바뀜



setting = {'vlaue': 5, 'value2':10, 'seq': [1,2,3,4,5]}
hash_key = hashlib.sha1(str(setting).encode()).hexdigest()[:6]
print(hash_key)

import pandas as pd
import numpy as np
df = pd.DataFrame({'a' : [1,2,3,4,5],'b' : [2,3,4,5,6]})

list(df.a)[0]


a= [1,2,3,4,5,2,10,7]
a.index(max(a))





import LSTM_MODEL_DATASET as LSTMMD

trainset = LSTMMD.StockDataset('^KS11', 7, 1, (2000, 1, 1), (2013, 1, 1))
for i in trainset:
    print(i)
    break

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import pandas_datareader.data as pdr
import datetime

class CV_Data_Spliter:
    def __init__(self, symbol, data_start, data_end,n_splits,test_size,gap=0):
        self.symbol = symbol
        self.start = datetime.datetime(*data_start)
        self.end = datetime.datetime(*data_end)
        self.data = pdr.DataReader(self.symbol, 'yahoo', self.start, self.end)
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        print(self.data.isna().sum())

        self.tscv = TimeSeriesSplit(gap=self.gap, max_train_size=None, n_splits=self.n_splits, test_size=self.test_size)

    def ts_cv_List(self):
        list = []
        for train_index, test_index in self.tscv.split(self.data):
            X_train, X_test = self.data.iloc[train_index, :], self.data.iloc[test_index,:]
            list.append((X_train, X_test))
        return list

    def __len__(self):
        return self.n_splits

    def __getitem__(self, item):
        datalist = self.ts_cv_List()
        return datalist[item]

print(tscv)

tscv.split(trainset[0],trainset[1])


data_start = (2010, 1, 1)
data_end = (2020, 12, 31)
cvds = CV_Data_Spliter('^KS11',data_start,data_end,n_splits=7,test_size=300)


cvds[0]




data = pdr.DataReader('^KS11', 'yahoo', datetime.datetime(*data_start), datetime.datetime(*data_end))

tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=7, test_size=300)

type(trainset)
a=np.array([1,2,3,4])
print(a)

for train_index, test_index in tscv.split(data):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data.iloc[train_index,:], data.iloc[test_index]




X = np.array([[11, 12], [13, 14], [15, 16], [17, 18], [19, 110],
              [21, 22], [23, 24], [25, 26], [27, 28], [29, 210],
              [31, 32], [33, 34], [35, 36], [37, 38], [39, 310],
              [41, 42], [43, 44], [45, 46], [47, 48], [49, 410],
              [51, 52], [53, 54], [55, 56], [57, 58], [59, 510],
              [61, 62], [63, 64], [65, 66], [67, 68], [69, 610],
              [71, 72], [73, 74], [75, 76], [77, 78], [79, 710],
              [81, 82], [83, 84], [85, 86], [87, 88], [89, 810],
              [91, 92], [93, 94], [95, 96], [97, 98], [99, 910]])
y = np.array([1, 2, 3, 4, 5, 6,
              7, 8, 9, 10, 11, 12,
              13, 14, 15, 16, 17, 18,
              19, 20, 21, 22, 23, 24,
              25, 26, 27, 28, 29, 30,
              31, 32, 33, 34, 35, 36,
              37, 38, 39, 40, 41, 42,
              43, 44, 45, 46, 47, 48,
              49, 50, 51, 52, 53, 54])
tscv = TimeSeriesSplit()


print(tscv)
tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=1)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    print(type(train_index))
    break

    X_train, X_test = trainset[train_index], trainset[test_index]



# TRAIN: [0] TEST: [1]
# TRAIN: [0 1] TEST: [2]
# TRAIN: [0 1 2] TEST: [3]
# TRAIN: [0 1 2 3] TEST: [4]
# TRAIN: [0 1 2 3 4] TEST: [5]
# Fix test_size to 2 with 12 samples
X = np.random.randn(12, 2)
y = np.random.randint(0, 2, 12)
tscv = TimeSeriesSplit(n_splits=3, test_size=2)
for train_index, test_index in tscv.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
# TRAIN: [0 1 2 3 4 5] TEST: [6 7]
# TRAIN: [0 1 2 3 4 5 6 7] TEST: [8 9]
# TRAIN: [0 1 2 3 4 5 6 7 8 9] TEST: [10 11]
# >>> # Add in a 2 period gap
tscv = TimeSeriesSplit(n_splits=3, test_size=2, gap=2)
for train_index, test_index in tscv.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
# TRAIN: [0 1 2 3] TEST: [6 7]
# TRAIN: [0 1 2 3 4 5] TEST: [8 9]
# TRAIN: [0 1 2 3 4 5 6 7] TEST: [10 11]

import numpy as np
import pandas as pd
ar = np.array([[1,2],
               [3,4],
               [5,6]])
dff = pd.DataFrame(ar)


df = pd.DataFrame(ar,columns=['a','b'])

c = df['a']
d = df[['a']]
type(c)
type(d)
