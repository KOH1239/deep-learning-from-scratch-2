#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertConfig, BertForSequenceClassification

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

# model generate, optimizer and criterion setting

# config = BertConfig.from_json_file('config.json')
config = BertConfig.from_pretrained('cl-tohoku/bert-base-japanese')
config.num_labels = 9
net = BertForSequenceClassification(config=config).to(device)
net.load_state_dict(torch.load(argvs[1]))

# Eval

real_data_num, ok = 0, 0
net.eval()
with torch.no_grad():
    for i in range(len(xtest)):
        x = torch.LongTensor(xtest[i]).unsqueeze(0).to(device)
        out = net(x)
        ans = out.logits
        ans1 = torch.argmax(ans,dim=1).item()
        if (ans1 == ytest[i]):
            ok += 1
        real_data_num += 1
print(ok, real_data_num, ok/real_data_num)
