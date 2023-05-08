#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML 
@File ：transformer.py
@Author ：Haopeng Wang
@Date ：2023-04-30 9:09 p.m. 
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
import math, copy, re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
print(torch.__version__)

class Embedding(nn.Module):
    def __int__(self, vocab_size, embed_dim):
        '''

        :param vocab_size:
        :param embed_dim:
        :return:
        '''
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        '''

        :param x:
        :return:
        '''
        out = self.embed(x)
        return out

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        '''

        :param max_seq_len:
        :param embed_model_dim:
        '''
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim,2):
                pe[pos,i]=math.sin(pos/(10000)**((2*i)/self.embed_dim))
                pe[pos, i+1] = math.cos(pos/(10000)**((2*(i+1))/self.embed_dim))
        # add a new dim 0th
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self, x):
        x = x*math.sqrt(self.embed_dim)
        seq_len = x.size(1)
        x = x+torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads=8):
        '''

        :param embed_dim:
        :param n_heads:
        '''
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim/self.n_heads)

        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.n_heads*self.single_head_dim, self.embed_dim)

    def forward(self, key, query, value, mask=None):
        '''

        :param key:
        :param query:
        :param value:
        :param mask:
        :return:
        '''

        batch_size = key.size(0)
        seq_length = key.size(1)
        seq_length_query = query.size(1)

        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)

        # key, query and value multiply three matrices first
        k = self.key_matrix(key)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)


