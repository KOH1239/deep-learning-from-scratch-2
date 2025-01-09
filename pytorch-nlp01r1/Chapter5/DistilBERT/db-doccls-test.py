#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import DistilBertModel, DistilBertConfig

import numpy as np
import pickle
import sys

argvs = sys.argv
argc = len(argvs)

config = DistilBertConfig.from_pretrained('bandainamco-mirai/distilbert-base-japanese')
bert = DistilBertModel(config=config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data Setting

with open('xtest.pkl','br') as fr:
    xtest = pickle.load(fr)

with open('ytest.pkl','br') as fr:
    ytest = pickle.load(fr)

# Define model

class DocCls(nn.Module):
    def __init__(self,bert):
        super(DocCls, self).__init__()
        self.bert = bert
        self.cls=nn.Linear(768,9)
    def forward(self,x):
        bout = self.bert(x)
        bs = len(bout[0])
        h0 = [ bout[0][i][0] for i in range(bs)]
        h0 = torch.stack(h0,dim=0)
        return self.cls(h0)

# model generate, optimizer and criterion setting

net = DocCls(bert).to(device)
net.load_state_dict(torch.load(argvs[1]))

# Learn

real_data_num, ok = 0, 0
net.eval()
with torch.no_grad():
    for i in range(len(xtest)):
        x = torch.LongTensor(xtest[i]).unsqueeze(0).to(device)
        ans = net(x)
        ans1 = torch.argmax(ans,dim=1).item()
        if (ans1 == ytest[i]):
            ok += 1
        real_data_num += 1
print(ok, real_data_num, ok/real_data_num)
