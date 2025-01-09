#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import torch

dic = {}
with open("vocab.txt","r",encoding="utf-8") as f:
    vocab = f.read()
    for id, word in enumerate(vocab.split('\n')):
        dic[word] = int(id)


lab = {'名詞': 0, '助詞': 1, '形容詞': 2,
     '助動詞': 3, '補助記号': 4, '動詞': 5, '代名詞': 6,
     '接尾辞': 7, '副詞': 8, '形状詞': 9, '記号': 10,
     '連体詞': 11, '接頭辞': 12, '接続詞': 13,
     '感動詞': 14, '空白': 15}

xdata = []
ydata = []

with open('train50K.dat','r') as f:
    x = [ dic['[CLS]'] ]
    y = [ -1 ]
    line = f.readline()
    i = 0
    while line:
        line = line.rstrip()
        if (line == "　\t空白"):
            x.append(dic['[UNK]'])
            y.append(15)
            line = f.readline()
            continue
        if (line == "EOS"):
            x.append(dic['[SEP]'])
            y.append(-1)
            xdata.append(x)
            ydata.append(y)
            i += 1
            print(i)
            if (i == 5000):
                break
            x = [ dic['[CLS]'] ]
            y = [ -1 ]
            line = f.readline()
            continue
        w, pos = line.split()
        if (w in dic):
            x.append(dic[w])
        else:
            x.append(dic['[UNK]'])
        y.append(lab[pos])
        line = f.readline()

with open('xtrain.pkl','bw') as fw:
    pickle.dump(xdata,fw)

with open('ytrain.pkl','bw') as fw:
    pickle.dump(ydata,fw)
