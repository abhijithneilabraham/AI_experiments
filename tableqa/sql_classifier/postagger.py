#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 21:00:49 2020

@author: abhijithneilabraham
"""

from nltk import word_tokenize,pos_tag,sent_tokenize
text = sent_tokenize("how many people died of cancer")
text=["how many","people","died","of","cancer"]
print(pos_tag(text))