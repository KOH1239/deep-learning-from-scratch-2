#!/usr/bin/python
# -*- coding: utf-8 -*-

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

batch_size = 4
dataset = MyDataset(xdata,ydata)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)

# Define model

# bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')

config = BertConfig.from_json_file('config.json')
bert = BertModel.from_pretrained('pytorch_model.bin',config=config)

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
    outfile = "sb-doccls-" + str(ep) + ".model"
    torch.save(net.state_dict(),outfile)
