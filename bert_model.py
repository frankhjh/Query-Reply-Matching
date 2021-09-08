#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall,Precision
from keras_bert import get_pretrained,PretrainedList,get_checkpoint_paths
from keras_bert import load_trained_model_from_checkpoint


def load_base_model():
    
    model_path=get_pretrained(PretrainedList.chinese_base) # choose chinese
    paths=get_checkpoint_paths(model_path)

    base_model=load_trained_model_from_checkpoint(paths.config,
                                                  paths.checkpoint,
                                                  training=True,
                                                  trainable=True,
                                                  seq_len=64)
    for l in base_model.layers:
        l.trainable=True
    print('bert model loaded!')
    return base_model

def create_bert_model():
    base_model=load_base_model()

    # add some layers 
    inputs=base_model.inputs[:2] # no position embedding
    x=base_model.layers[-7].output
    x=Lambda(lambda x:x[:,0])(x) # [CLS]
    x=Dense(units=128,activation=K.tanh)(x)
    x=Dropout(0.2)(x)
    outputs=Dense(1,activation='sigmoid')(x)

    bert=Model(inputs,outputs)
    bert.compile(optimizer=Adam(1e-5),
                 loss='binary_crossentropy',
                 metrics=[Precision(),Recall()])
    return bert











