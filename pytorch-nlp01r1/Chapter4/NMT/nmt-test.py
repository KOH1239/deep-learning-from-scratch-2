#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import sys

argvs = sys.argv
argc = len(argvs)

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
with open('test.en','r',encoding='utf-8') as f:
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
with open('test.ja','r',encoding='utf-8') as f:
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

class MyNMT(nn.Module):
    def __init__(self, jv, ev, k):
        super(MyNMT, self).__init__()
        self.jemb = nn.Embedding(jv, k)
        self.eemb = nn.Embedding(ev, k)
        self.lstm1 = nn.LSTM(k, k, num_layers=2,
                             batch_first=True)
        self.lstm2 = nn.LSTM(k, k, num_layers=2,
                             batch_first=True)
        self.W = nn.Linear(k, ev)
    def forward(self, jline, eline):
        x = self.jemb(jline)
        ox, (hnx, cnx) = self.lstm1(x)
        y = self.eemb(eline)
        oy, (hny, cny) = self.lstm2(y,(hnx, cnx))
        out = self.W(oy)
        return out

# model generate, optimizer and criterion setting

demb = 200
net = MyNMT(jv, ev, demb).to(device)

net.load_state_dict(torch.load(argvs[1]))

# Eval

esid = ew2id['<s>']
eeid = ew2id['</s>']
net.eval()
with torch.no_grad():
    for i in range(len(jdata)):
        jinput = torch.LongTensor([jdata[i][1:]]).to(device)
        x = net.jemb(jinput)
        ox, (hn, cn) = net.lstm1(x)
        wid = esid
        sl = 0
        while True:
            wids = torch.LongTensor([[wid]]).to(device)
            y = net.eemb(wids)
            oy, (hn, cn) = net.lstm2(y,(hn, cn))
            oy1 = net.W(oy)
            wid = torch.argmax(F.softmax(oy1[0],dim=1)).item()
            if (wid == eeid):
                break
            print(eid2w[wid]," ",end='')
            sl += 1
            if (sl == 30):
                break
        print()
