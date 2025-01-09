#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

###################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##################

# Data setting

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

###########################

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

batch_size = 100
dataset = MyDataset(jdata,edata)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)


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
        return self.W(oy2)

# model generate, optimizer and criterion setting

demb = 200
net = MyAttNMT(jv, ev, demb).to(device)
optimizer = optim.SGD(net.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=-1)

# Learn

net.train()
for ep in range(20):
    i = 0
    for xs, ys in dataloader:
        xs1, ys1, ys2 = [], [], []
        for k in range(len(xs)):
            tid = xs[k]
            xs1.append(torch.LongTensor(tid[1:]))
            tid = ys[k]
            ys1.append(torch.LongTensor(tid[:-1]))
            ys2.append(torch.LongTensor(tid[1:]))
        jinput = pad_sequence(xs1, batch_first=True).to(device)
        einput = pad_sequence(ys1, batch_first=True).to(device)
        gans = pad_sequence(ys2, batch_first=True, padding_value=-1.0).to(device)
        out = net(jinput, einput)
        loss = criterion(out[0],gans[0])
        for h in range(1,len(gans)):
            loss += criterion(out[h],gans[h])
        print(ep, i, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
    outfile = "attnmt2-" + str(ep) + ".model"
    torch.save(net.state_dict(),outfile)
