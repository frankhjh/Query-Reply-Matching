#!/usr/bin/env python
from tensorflow.keras.callbacks import ModelCheckpoint
from preprocess import df_train
from prepare_data import build_tokenizer,load_data
from bert_model import load_base_model,create_bert_model
import argparse
import os
import json

parser=argparse.ArgumentParser(description='training parameters')
parser.add_argument('--epochs',type=int,default=15)
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--validation_split',type=float,default=0.1)
args=parser.parse_args()

def train(epochs,batch_size,validation_split):
    # load data
    train_x,train_y=load_data(df_train,max_len=64,label=True)
    # int->float
    train_y=train_y.astype(float)
    print('>>data loaded.')
    # model checkpoint
    base_path='./tmp/checkpoint'
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    checkpointer=ModelCheckpoint(filepath=os.path.join(base_path,'{epoch:02d}.ckpt'),
                                 save_best_only=False,
                                 verbose=1,
                                 save_weights_only=True)
    # load pre-trained model
    bert=create_bert_model()
    print('>>model loaded.')
    # train
    history=bert.fit(train_x,
                     train_y,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_split=validation_split,
                     callbacks=[checkpointer])
    
    # save precision & recall result for each epoch
    val_precision=history.history['val_precision']
    val_recall=history.history['val_recall']
    dic={'val_precision':val_precision,'val_recall':val_recall}
    
    with open('./tmp/metric_scores.json','w') as f:
        json.dump(dic,f)
    
    return history

if __name__=='__main__':
    # parameters
    epochs=args.epochs
    batch_size=args.batch_size
    validation_split=args.validation_split
    # train
    train(epochs,batch_size,validation_split)






