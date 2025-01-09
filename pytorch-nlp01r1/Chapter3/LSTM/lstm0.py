#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import pickle

###################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##################

with open('dic.pkl','br') as f:
    dic = pickle.load(f)

labels = {'名詞': 0, '助詞': 1, '形容詞': 2,
     '助動詞': 3, '補助記号': 4, '動詞': 5, '代名詞': 6,
     '接尾辞': 7, '副詞': 8, '形状詞': 9, '記号': 10,
     '連体詞': 11, '接頭辞': 12, '接続詞': 13,
     '感動詞': 14, '空白': 15}

# Data setting

with open('xtrain.pkl','br') as f:
    xtrain = pickle.load(f)

with open('ytrain.pkl','br') as f:
    ytrain = pickle.load(f)

# Define model

class MyLSTM(nn.Module):
    def __init__(self, vocsize, posn, hdim):
        super(MyLSTM, self).__init__()
        self.embd = nn.Embedding(vocsize, hdim)
        self.lstm = nn.LSTM(input_size=hdim, hidden_size=hdim)
        self.ln   = nn.Linear(hdim, posn)
    def forward(self, x):
        x = self.embd(x)
        lo, (hn, cn) = self.lstm(x)
        out = self.ln(lo)
        return out

# model generate, optimizer and criterion setting

net = MyLSTM(len(dic)+1, len(labels), 100).to(device)
optimizer = optim.SGD(net.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()

# Learn

for ep in range(1,11):
    loss1K = 0.0
    for i in range(len(xtrain)):
        x = [ xtrain[i] ]
        x = torch.LongTensor(x).to(device)
        output = net(x)
        y = torch.LongTensor( ytrain[i] ).to(device)
        loss = criterion(output[0],y)
        if (i % 1000 == 0):
            print(i, loss1K)
            loss1K = loss.item()
        else:
            loss1K += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    outfile = "lstm0-" + str(ep) + ".model"
    torch.save(net.state_dict(),outfile)
