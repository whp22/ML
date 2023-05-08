#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML 
@File ：nn_embedding.py
@Author ：Haopeng Wang
@Date ：2023-05-03 4:13 p.m. 
'''
import torch
import torch.nn as nn
import torchtext
from torchnlp import *
import os
import collections

os.makedirs('./data', exist_ok=True)
train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='./data')
classes = ['World', 'Sports', 'Business', "Sci/Tech"]
print(list(train_dataset)[0])

embedding = nn.Embedding(10,50)
test = embedding(torch.LongTensor([0]))
print(test)