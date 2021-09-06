#!/usr/bin/env python
from keras_bert import Tokenizer,get_pretrained,PretrainedList,get_checkpoint_paths
import os
import codecs
import numpy as np
from tqdm import tqdm



def build_tokenizer():
    
    model_path=get_pretrained(PretrainedList.chinese_base) # choose chinese
    paths=get_checkpoint_paths(model_path)

    #token
    token_dict={}
    with codecs.open(paths.vocab,'r','utf8') as reader:
        for line in reader:
            token=line.strip()
            token_dict[token]=len(token_dict)

    tokenizer=Tokenizer(token_dict)
    return tokenizer


#load data
def load_data(df,max_len,label=True):
    #global tokenizer
    tokenizer=build_tokenizer()
    
    indices,segments=[],[]
    if label:
        labels=df['label'].tolist()
        for i in tqdm(range(len(df))):
            id,segment=tokenizer.encode(first=df['query_content'][i],second=df['reply_content'][i],max_len=max_len)
            indices.append(id)
            segments.append(segment)
        items=list(zip(indices,segments,labels))
        np.random.shuffle(items)
        indices,segments,labels=zip(*items)
    
        indices=np.array(indices)
        segments=np.array(segments)
        labels=np.array(labels)
        
        return [indices,segments],labels
    else:
        for i in tqdm(range(len(df))):
            id,segment=tokenizer.encode(first=df['query_content'][i],second=df['reply_content'][i],max_len=max_len)
            indices.append(id)
            segments.append(segment)
        
        indices=np.array(indices)
        segments=np.array(segments)
        return [indices,segments]



