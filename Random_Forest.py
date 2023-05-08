#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML 
@File ：Random_Forest.py
@Author ：Haopeng Wang
@Date ：2023-05-07 8:56 p.m. 
'''
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
np.random.seed(0)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['species']=pd.Categorical.from_codes(iris.target, iris.target_names)

df['is_train'] = np.random.uniform(0,1,len(df)) <= .75

train, test = df[df['is_train']==True], df[df['is_train']==False]
# print(len(train), len(test))

features = df.columns[:4]

y = pd.factorize(train['species'])[0]

clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(train[features],y)
clf.predict(test[features])

preds = iris.target_names[clf.predict(test[features])]

print(preds[0:5])
