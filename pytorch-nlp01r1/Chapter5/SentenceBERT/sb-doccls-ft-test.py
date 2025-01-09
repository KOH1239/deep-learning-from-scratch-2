#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertConfig

import numpy as np
import pickle
import sys

argvs = sys.argv
argc = len(argvs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data Setting

with open('xtest.pkl','br') as fr:
    xtest = pickle.load(fr)

with open('ytest.pkl','br') as fr:
    ytest = pickle.load(fr)

# Define model

config = BertConfig.from_json_file('config.json')
bert = BertModel(config=config)

class DocCls(nn.Module):
    def __init__(self,bert):
        super(DocCls, self).__init__()
        self.bert = bert
        self.cls=nn.Linear(768,9)
    def forward(self,x1,x2):
        bout = self.bert(input_ids=x1, attention_mask=x2)
        a = self.mean_pooling(bout, x2)
        return self.cls(a)
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

# model generate, optimizer and criterion setting

net = DocCls(bert).to(device)
net.load_state_dict(torch.load(argvs[1]))

# Learn

real_data_num, ok = 0, 0
net.eval()
with torch.no_grad():
    for i in range(len(xtest)):
        x = torch.LongTensor(xtest[i]).unsqueeze(0).to(device)
        xm =  torch.LongTensor([ [1] * len(xtest[i]) ]).to(device)
        ans = net(x,xm)
        ans1 = torch.argmax(ans,dim=1).item()
        if (ans1 == ytest[i]):
            ok += 1
        real_data_num += 1
print(ok, real_data_num, ok/real_data_num)
