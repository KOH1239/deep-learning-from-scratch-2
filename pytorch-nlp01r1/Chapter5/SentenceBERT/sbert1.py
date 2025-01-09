#!/usr/bin/python
# -*- coding: sjis -*-

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertConfig
from transformers import BertJapaneseTokenizer
from transformers.models.bert_japanese import tokenization_bert_japanese

tknz = BertJapaneseTokenizer(vocab_file='vocab.txt', do_lower_case=False,do_basic_tokenize=False)
tknz.word_tokenizer =  tokenization_bert_japanese.MecabTokenizer()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = BertConfig.from_json_file('config.json')
# net = BertModel.from_pretrained('pytorch_model.bin',config=config).to(device)
net = BertModel.from_pretrained('cl-tohoku/bert-base-japanese').to(device)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

ss = [ "大学構内では喫煙禁止です。",
       "学校でタバコを吸うのはダメです。",
       "今日は学校でタバコを買った。"]


xs1, xmsk = [], []
for i in range(len(ss)):
    tid = tknz.encode(ss[i])
    xs1.append(torch.LongTensor(tid))
    xmsk.append(torch.LongTensor([1] * len(tid)))
xs1 = pad_sequence(xs1, batch_first=True).to(device)
xmsk = pad_sequence(xmsk, batch_first=True).to(device)

out = net(xs1,xmsk)
sv = mean_pooling(out, xmsk)
print("cos(s0,s1) = ", torch.cosine_similarity(sv[0], sv[1], dim=0).item())
print("cos(s0,s2) = ", torch.cosine_similarity(sv[0], sv[2], dim=0).item())
print("cos(s1,s2) = ", torch.cosine_similarity(sv[1], sv[2], dim=0).item())


