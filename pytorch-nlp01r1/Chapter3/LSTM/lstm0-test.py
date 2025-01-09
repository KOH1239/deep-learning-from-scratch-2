#!/usr/bin/python
# -*- coding: sjis -*-

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import pickle

argvs = sys.argv
argc = len(argvs)

###################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##################

with open('dic.pkl','br') as f:
    dic = pickle.load(f)

with open('label.pkl','br') as f:
    lab = pickle.load(f)

# Data setting

with open('xtest.pkl','br') as f:
    xtest = pickle.load(f)

with open('ytest.pkl','br') as f:
    ytest = pickle.load(f)

# Define model

class MyLSTM(nn.Module):
    def __init__(self, vocsize, posn, hdim):
        super(MyLSTM, self).__init__()
        self.embd = nn.Embedding(vocsize, hdim)
        self.lstm = nn.LSTM(input_size=hdim, hidden_size=hdim, batch_first=True)
        self.ln   = nn.Linear(hdim, posn)
    def forward(self, x):
        x = self.embd(x)
        lo, (hn, cn) = self.lstm(x)
        out = self.ln(lo)
        return out

# model generate, optimizer and criterion setting

net = MyLSTM(len(dic)+1, len(lab), 100).to(device)
net.load_state_dict(torch.load(argvs[1]))

# Eval

real_data_num = 0  # データの個数
net.eval()
with torch.no_grad():
    ok = 0  # ok は正解数
    for i in range(len(xtest)):
        real_data_num += len(xtest[i])
        x = [ xtest[i] ]
        x = torch.LongTensor(x).to(device)
        output = net(x)
        ans = torch.argmax(output[0],dim=1)
        y = torch.LongTensor(ytest[i]).to(device)
        ok += torch.sum(ans == y).item()
    print(ok, real_data_num, ok/real_data_num)
