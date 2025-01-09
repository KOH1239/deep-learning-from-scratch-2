#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification
import numpy as np
import pickle

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

batch_size = 3
dataset = MyDataset(xdata,ydata)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)

# Define model

net = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese', num_labels = 9).to(device)
optimizer = optim.SGD(net.parameters(),lr=0.001)

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
        ys = torch.LongTensor(ys).to(device)
        out = net(xs1,attention_mask=xmsk,labels=ys)
        loss = out.loss
        lossK += loss.item()
        if (i % 10 == 0):
            print(ep, i, lossK)
            lossK = 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
    outfile = "doccls4-" + str(ep) + ".model"
    torch.save(net.state_dict(),outfile)
