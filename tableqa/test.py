#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:57:06 2020

@author: abhijithneilabraham
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine,types
from pandas.io.sql import SQLTable
import column_types
import ast
import json
import os
path='cleaned_data/Activities - Data.csv'

data_dir="cleaned_data"
import sqlite3
def get_schema_for_csv(csv_path):
    try:
        with open(os.path.join('schema', csv_path[len(data_dir) + 1:-4]) + '.json', 'r') as f:
            return json.load(f)
    except:
        return None
def get_dataframe(csv_path):
    return pd.read_csv(csv_path)



def csv2sql(data_frame, schema, output_path):
    engine = create_engine('sqlite://', echo=False)
    data_frame = data_frame.fillna(data_frame.mean())
    sql_schema = {}
    for col in schema['columns']:
        colname = col['name']
        coltype = col['type']
        coltype = column_types.get(coltype).sql_type
        if '(' in coltype:
            coltype, arg = coltype.split('(')
            arg ='(' + arg[:-1] + ',)'
            coltype = getattr(types, coltype)(*(ast.literal_eval(arg)))
        else:
            coltype = getattr(types, coltype)()
        sql_schema[colname] = coltype
    data_frame.to_sql(schema['name'].lower(), con=engine, if_exists='replace', dtype=sql_schema)
    print("Dumping Table",schema["name"])
    conn=engine.connect()

    with open(output_path+'.sql', 'w') as stream:
        for line in conn.connection.iterdump():
            stream.write(line)
            stream.write('\n')

    

    

df=get_dataframe(path)
schema=get_schema_for_csv(path)
csv2sql(df,schema,"abcd")