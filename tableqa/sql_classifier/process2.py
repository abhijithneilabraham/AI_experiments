#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 04:11:18 2020

@author: abhijithneilabraham
"""


import jsonlines
import pandas as pd

data=pd.DataFrame()     
questions=[]
types=[]
with jsonlines.open('train.jsonl') as f:
    for line in f.iter():
        questions.append(line["question"])
        types.append(line["sql"]["agg"])
        
        
data["questions"],data["types"]=questions,types
data.to_csv("wikidata.csv")
        
        
        
   

        