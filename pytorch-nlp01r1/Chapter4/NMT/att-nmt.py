#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data setting

id, eid2w, ew2id = 1, {}, {}
with open('train.en.vocab.4k','r',encoding='utf-8') as f:
    for w in f:
        w = w.strip()
        eid2w[id] = w
        ew2id[w] = id
        id += 1
ev = id

edata = []
with open('train.en','r',encoding='utf-8') as f:
    for sen in f:
        wl = [ew2id['<s>']]
        for w in sen.strip().split():
            if w in ew2id:
                wl.append(ew2id[w])
            else:
                wl.append(ew2id['<unk>'])
        wl.append(ew2id['</s>'])
        edata.append(wl)

id, jid2w, jw2id = 1, {}, {}
with open('train.ja.vocab.4k','r',encoding='utf-8') as f:
    id = 1
    for w in f:
        w = w.strip()
        jid2w[id] = w
        jw2id[w] = id
        id += 1
jv = id

jdata = []
with open('train.ja','r',encoding='utf-8') as f:
    for sen in f:
        wl = [jw2id['<s>']]
        for w in sen.strip().split():
            if w in jw2id:
                wl.append(jw2id[w])
            else:
                wl.append(jw2id['<unk>'])
        wl.append(jw2id['</s>'])
        jdata.append(wl)

# Define model

class MyAttNMT(nn.Module):
    def __init__(self, jv, ev, k):
        super(MyAttNMT, self).__init__()
        self.jemb = nn.Embedding(jv, k)
        self.eemb = nn.Embedding(ev, k)
        self.lstm1 = nn.LSTM(k, k, num_layers=2,
                             batch_first=True)
        self.lstm2 = nn.LSTM(k, k, num_layers=2,
                             batch_first=True)
        self.Wc = nn.Linear(2*k, k)
        self.W = nn.Linear(k, ev)
    def forward(self, jline, eline):
        x = self.jemb(jline)
        ox, (hnx, cnx) = self.lstm1(x)
        y = self.eemb(eline)
        oy, (hny, cny) = self.lstm2(y,(hnx, cnx))
        ox1 = ox.permute(0,2,1)
        sim = torch.bmm(oy,ox1)
        bs, yws, xws = sim.shape
        sim2 = sim.reshape(bs*yws,xws)
        alpha = F.softmax(sim2,dim=1).reshape(bs, yws, xws)
        ct = torch.bmm(alpha,ox)
        oy1 = torch.cat([ct,oy],dim=2)
        oy2 = self.Wc(oy1)
        return self.W(oy2)

# model generate, optimizer and criterion setting

demb = 200
net = MyAttNMT(jv, ev, demb).to(device)
optimizer = optim.SGD(net.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()

# Learn

for epoch in range(20):
    loss1K = 0.0
    for i in range(len(jdata)):
        jinput = torch.LongTensor([jdata[i][1:]]).to(device)
        einput = torch.LongTensor([edata[i][:-1]]).to(device)
        out = net(jinput, einput)
        gans = torch.LongTensor([edata[i][1:]]).to(device)
        loss = criterion(out[0],gans[0])
        loss1K += loss.item()
        if (i % 100 == 0):
            print(epoch, i, loss1K)
            loss1K = 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    outfile = "attnmt-" + str(epoch) + ".model"
    torch.save(net.state_dict(),outfile)
