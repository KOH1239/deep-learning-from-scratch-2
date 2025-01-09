#!/usr/bin/python
# -*- coding: sjis -*-

from transformers import AutoTokenizer
import pickle
import re

tknz = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

xdata, ydata = [],[]
with open('test.tsv','r',encoding='utf-8') as f:
    for line in f:
        line = line.rstrip()
        result = re.match('^(\d+)\t(.+?)$', line)
        ydata.append(int(result.group(1)))
        sen = result.group(2)
        tid = tknz.encode(sen)
        if (len(tid) > 512):  # 最大長は 512
            tid = tid[:512]
        xdata.append(tid)

with open('xtest.pkl','bw') as fw:
    pickle.dump(xdata,fw)

with open('ytest.pkl','bw') as fw:
    pickle.dump(ydata,fw)
