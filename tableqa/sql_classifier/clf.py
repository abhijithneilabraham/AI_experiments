#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 22:30:11 2020

@author: abhijithneilabraham
"""

import pandas as pd
import numpy as np
from numpy import asarray
from nltk.tokenize import sent_tokenize
np.random.seed(7)
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from tensorflow.keras.models import Sequential, load_model, model_from_config
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from sklearn import preprocessing
import pickle


def get_keras_model(len_classes):
    """Define the model."""
    bert_model = Sequential()
    bert_model.add(Dense(300, input_shape=[768,], activation='relu'))
    bert_model.add(Dense(64,activation='relu'))
    bert_model.add(Dense(len_classes, activation='softmax'))
    bert_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    bert_model.summary()

    return bert_model

data=pd.read_csv("dataset.csv",usecols=["Questions","Category1"])
le = preprocessing.LabelEncoder()
categories=data["Category1"].tolist()
encoded_cat=le.fit_transform(categories)
with open('transform.pkl', 'wb') as output:
    pickle.dump(le, output)
x_train, x_test, y_train,y_test =train_test_split(data["Questions"],encoded_cat)

bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

train_embeddings = asarray(bert_model.encode(x_train.tolist()), dtype = "float32")
test_embeddings = asarray(bert_model.encode(x_test.tolist()), dtype = "float32")
y_train=asarray(y_train,dtype="float32")
y_test=asarray(y_test,dtype="float32")

model = get_keras_model(len(le.classes_))
model.fit(train_embeddings, y_train, epochs=50,validation_data=(test_embeddings,y_test))
print(le.classes_)
print('\n# Evaluate on test data')
results = model.evaluate(test_embeddings, y_test, batch_size=128)
print('test loss, test acc:', results)



model.save("Question_Classifier.h5")


        

    



    

