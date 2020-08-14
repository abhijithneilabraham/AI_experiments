#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 23:02:31 2020

@author: abhijithneilabraham
"""

import pandas as pd
import numpy as np
data=pd.read_csv("Question_Classification_Dataset.csv",usecols=["Questions","Category1"])
y=[]
for i in data["Category1"]:
    if i =="NUM":
        y.append(0)        
    else:
        y.append(1)
data["type"]=y
print(data)  
data.to_csv("refined_data.csv")      
