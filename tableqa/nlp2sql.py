#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 00:23:52 2020

@author: abhijithneilabraham
"""
from eywa.nn import NNClassifier
docs=['which are','what are','how many','sum of',"total","find all","search for"]
labels=['find','find','count','count','count','find','find']
nnclf=NNClassifier(docs, labels)
print(nnclf.predict("how many people died of stomach cancer in year 2011"))