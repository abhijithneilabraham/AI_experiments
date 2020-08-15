#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 23:02:31 2020

@author: abhijithneilabraham
"""

import pandas as pd
import numpy as np
data=pd.read_csv("refined_data.csv",usecols=["Questions","Category1"])[:400]
y=[]
for i in data["Category1"]:
    if i =="NUM":
        y.append("count")        
    else:
        y.append("desc")
data["Category1"]=y

data2=pd.read_csv("maxminavg.csv",usecols=["Questions","Category1"])
dataset=pd.concat([data,data2])
dataset.reset_index(drop=True,inplace=True)
dataset.to_csv("dataset.csv")      
