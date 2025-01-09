#!/usr/bin/python
# -*- coding: sjis -*-

import glob
import random

dira = ["dokujo-tsushin", "it-life-hack", "kaden-channel", "livedoor-homme",
        "movie-enter", "peachy", "smax", "sports-watch", "topic-news"]

alldata = []
for cls, dir in enumerate(dira):
    files = glob.glob(dir + "/*")
    for file in files:
        with open(file,'r',encoding='utf-8') as f:
            data = f.read().split('\n')
            text = str(cls) + "\t" 
            for i in range(3,len(data)):
                text += data[i]
            alldata.append(text)

random.shuffle(alldata)
train = alldata[:738]
test = alldata[738:1476]

with open('train.tsv','w',encoding='utf-8') as f:
    for i in range(738):
        f.write(alldata[i])
        f.write('\n')
        
with open('test.tsv','w',encoding='utf-8') as f:
    for i in range(738,1476):
        f.write(alldata[i])
        f.write('\n')        



            
