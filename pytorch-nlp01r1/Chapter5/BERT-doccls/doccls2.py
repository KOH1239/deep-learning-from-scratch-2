#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel

import numpy as np
import pickle
import sys

argvs = sys.argv
argc = len(argvs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DataLoader

class MyDataset(Dataset):
    def __init__(self, xdata, ydata):
        self.data = xdata
        self.label = ydata
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return (x,y)

def my_collate_fn(batch):
    images, targets= list(zip(*batch))
    xs = list(images)
    ys = list(targets)
    return xs, ys

with open('xtrain.pkl','br') as fr:
    xdata = pickle.load(fr)

with open('ytrain.pkl','br') as fr:
    ydata = pickle.load(fr)

batch_size = 2
dataset = MyDataset(xdata,ydata)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, collate_fn=my_collate_fn)

# Define model

bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')

class DocCls(nn.Module):
    def __init__(self,bert):
        super(DocCls, self).__init__()
        self.bert = bert
        self.cls=nn.Linear(768,9)
    def forward(self,x1,x2):
        bout = self.bert(input_ids=x1, attention_mask=x2)
        bs = len(bout[0])
        h0 = [ bout[0][i][0] for i in range(bs)]
        h0 = torch.stack(h0,dim=0)
        return self.cls(h0)

# model generate, optimizer and criterion setting

net = DocCls(bert).to(device)
optimizer = optim.SGD(net.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

# Learn

net.train()
for ep in range(30):
    i, lossK = 0, 0.0
    for xs, ys in dataloader:
        xs1, xmsk = [], []
        for k in range(len(xs)):
            tid = xs[k]
            xs1.append(torch.LongTensor(tid))
            xmsk.append(torch.LongTensor([1] * len(tid)))
        xs1 = pad_sequence(xs1, batch_first=True).to(device)
        xmsk = pad_sequence(xmsk, batch_first=True).to(device)
        outputs = net(xs1,xmsk)
        ys = torch.LongTensor(ys).to(device)
        loss = criterion(outputs, ys)
        lossK += loss.item()
        if (i % 10 == 0):
            print(ep, i, lossK)
            lossK = 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
    outfile = "doccls2-" + str(ep) + ".model"
    torch.save(net.state_dict(),outfile)
