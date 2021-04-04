## 가상의 데이터셋
import numpy as np
from torch.utils.data import Dataset, DataLoader

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
print(tscv)

tscv.split(trainset[0],trainset[1])

data_start = (2000, 1, 1)
data_end = (2020, 12, 31)
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