#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import sys

argvs = sys.argv
argc = len(argvs)

###################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##################

# Data setting

id, eid2w, ew2id = 1, {}, {}
with open('train.en.vocab.4k','r') as f:
    for w in f:
        w = w.strip()
        eid2w[id] = w
        ew2id[w] = id
        id += 1

ev = id

edata = []
with open('test.en','r') as f:
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
with open('train.ja.vocab.4k','r') as f:
    id = 1
    for w in f:
        w = w.strip()
        jid2w[id] = w
        jw2id[w] = id
        id += 1
jv = id

jdata = []
with open('test.ja','r') as f:
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
        self.jemb = nn.Embedding(jv, k, padding_idx=0)
        self.eemb = nn.Embedding(ev, k, padding_idx=0)
        self.lstm1 = nn.LSTM(k, k, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(k, k, num_layers=2, batch_first=True)
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
        return torch.tanh(self.W(oy2))

# model generate, optimizer and criterion setting

demb = 200
net = MyAttNMT(jv, ev, demb).to(device)

net.load_state_dict(torch.load(argvs[1]))

# Learn

esid = ew2id['<s>']
eeid = ew2id['</s>']
net.eval()
with torch.no_grad():
    for i in range(len(jdata)):
        jinput = torch.LongTensor([ jdata[i][1:] ]).to(device)
        x = net.jemb(jinput)
        ox, (hnx, cnx) = net.lstm1(x)
        wid = esid
        sl = 0
        while True:
            wids = torch.LongTensor([[ wid ]]).to(device)
            y = net.eemb(wids)
            oy, (hnx, cnx) = net.lstm2(y,(hnx, cnx))
            ox1 = ox.permute(0,2,1)
            sim = torch.bmm(oy,ox1)
            bs, yws, xws = sim.shape
            sim2 = sim.reshape(bs*yws,xws)
            alpha = F.softmax(sim2,dim=1).reshape(bs, yws, xws)
            ct = torch.bmm(alpha,ox)
            oy1 = torch.cat([ct,oy],dim=2)
            oy2 = net.Wc(oy1)
            oy3 = net.W(oy2)
            wid = torch.argmax(oy3[0]).item()
            if (wid == eeid):
                break
            print(eid2w[wid]," ",end='')
            sl += 1
            if (sl == 30):
                break
        print()
