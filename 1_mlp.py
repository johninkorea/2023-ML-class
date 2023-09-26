import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import adam
from sklearn import datasets
import sklearn

parse = argparse.ArgumentParser(description="asd")
parse.add_argument("--epoch", default=100, type=int)
parse.add_argument("--lr", default=.001, type=float)
parse.add_argument("--tr", default=.6, type=float)

param = parse.parse_args(args=[])

iris = datasets.load_iris()
# print(iris)

x=iris['data']
y=iris['target']

x_train, x_text, y_train, y_text = sklearn.train_test_split(x,y,test_size=1-param.tr)













