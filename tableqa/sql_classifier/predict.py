#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 00:27:17 2020

@author: abhijithneilabraham
"""

import pickle
with open('transform.pkl', 'rb') as le_pickle:
    le = pickle.load(le_pickle) 

from tensorflow.keras.models import Sequential, load_model, model_from_config
from sentence_transformers import SentenceTransformer
from numpy import asarray
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
model=load_model("Question_Classifier.h5")



def test():  
    while 1:        
        ip=input("Enter your question \n")
        if ip!="quit":
            emb=asarray(bert_model.encode(ip))
            print(le.inverse_transform(model.predict_classes(emb))[0])
        else:
            break
        
test()



