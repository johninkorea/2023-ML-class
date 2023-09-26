import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as opt
import torch.nn as nn

from sklearn import datasets
from sklearn.model_selection import train_test_split

parse = argparse.ArgumentParser(description="asd")
parse.add_argument("--epoch", default=100, type=int)
parse.add_argument("--lr", default=.001, type=float)
parse.add_argument("--tr", default=.6, type=float)

param = parse.parse_args(args=[])

iris = datasets.load_iris()
# print(iris)

x=iris['data']
y=iris['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1-param.tr)


x_train=torch.FloatTensor(x_train)
x_test=torch.FloatTensor(x_test)
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)


class Net(nn.Module):
    def __init__(self, in_ch, out_ch, num_node=[20,20]):
        super(Net, self).__init__()
    
        self.relu = nn.ReLU()
        self.sorftmax = nn.Softmax(dim=1)

        self.fc1 = nn.Linear(in_ch, num_node[0])
        self.fc2 = nn.Linear(num_node[0], num_node[1])
        self.fc3 = nn.Linear(num_node[0], out_ch)

    def forward(self, x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.sorftmax(self.sorftmax(x))
        return x


model=Net(in_ch=4, out_ch=3)
loss=nn.CrossEntropyLoss()
optmizer=opt.Adam(model.parameters(), lr=param.lr)


total_loss=[]
acc_train=[]
acc_test=[]

model.train() #강속 받은 함수
for epoch in range(param.epoch):
    optmizer.zero_grad()

    y_pred = model(x_train)
    losses = loss(y_pred, y_train)
    losses.backward()
    optmizer.step()

    total_loss.append(losses.data.item())

    with torch.no_grad:
        acc = (torch.argmax(y_pred, dim=1) == y_train).type(torch.FloatTensor)
        acc_train.append(acc.mean().data.item())







