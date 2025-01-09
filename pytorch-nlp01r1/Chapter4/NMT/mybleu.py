#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
argvs = sys.argv

gold = []
with open('test.en','r') as f:
    for sen in f:
        w = sen.strip().split()
        gold.append([ w ])

myans = []
with open(argvs[1],'r') as f:
    for sen in f:
        w = sen.strip().split()
        myans.append(w)

from nltk.translate.bleu_score import corpus_bleu
score = corpus_bleu(gold, myans)
print(100*score)
