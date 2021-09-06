#!/usr/bin/env python
import pandas as pd
import numpy as np
import re
from utils.url_replacer import replace_url

# load data
query_train=pd.read_table('./data/train/train.query.tsv',header=None)
reply_train=pd.read_table('./data/train/train.reply.tsv',header=None)
query_test=pd.read_table('./data/test/test.query.tsv',encoding='gbk',header=None)
reply_test=pd.read_table('./data/test/test.reply.tsv',encoding='gbk',header=None)

# processing of df
query_train.rename(columns={0:'query_id',1:'query_content'},inplace=True)
reply_train.rename(columns={0:'query_id',1:'reply_id',2:'reply_content',3:'label'},inplace=True)
df_train=pd.merge(reply_train,query_train,on='query_id',how='left')[['query_id','query_content','reply_content','label']]

query_test.rename(columns={0:'query_id',1:'query_content'},inplace=True)
reply_test.rename(columns={0:'query_id',1:'reply_id',2:'reply_content'},inplace=True)
df_test=pd.merge(reply_test,query_test,on='query_id',how='left')[['query_id','query_content','reply_content']]

# replace web address

df_train['query_content']=df_train['query_content'].apply(str).apply(replace_url)
df_train['reply_content']=df_train['reply_content'].apply(str).apply(replace_url)
df_test['query_content']=df_test['query_content'].apply(str).apply(replace_url)
df_test['reply_content']=df_test['reply_content'].apply(str).apply(replace_url)
