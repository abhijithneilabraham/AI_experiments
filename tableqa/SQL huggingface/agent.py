#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 03:40:16 2020

@author: abhijithneilabraham
"""

from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")

def get_sql(query):
  input_text = "translante English to SQL: %s </s>" % query
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'])

  return tokenizer.decode(output[0])

query = "who is the sister with maximum age in family data"

get_sql(query)