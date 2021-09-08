#!/usr/bin/env python
from preprocess import reply_test,df_test
from prepare_data import build_tokenizer,load_data
from bert_model import load_base_model,create_bert_model
from utils.metric_plot import plot_precision_recall_f1
import json

def predict():
    
    # load precision and recall stored in json
    with open('./tmp/metric_scores.json','r') as f:
        dic=json.load(f)
    
    val_precision=dic['val_precision']
    val_recall=dic['val_recall']
    epochs=len(val_precision)
    # image save path
    save_path='./tmp/metric_scores.png'

    # run the plot
    best_f1_epoch=plot_precision_recall_f1(epochs,val_precision,val_recall,save_path)

    # load the optimal model(use f1)
    best_f1_epoch=f'0{best_f1_epoch}' if best_f1_epoch<10 else f'{best_f1_epoch}'
    ck_path=f'./tmp/checkpoint/{best_f1_epoch}.ckpt'
    model=create_bert_model()
    model.load_weights(ck_path)
    
    # load test data
    test_x=load_data(df_test,max_len=64,label=False)

    # predict
    pred=model.predict(test_x,verbose=True)
    return pred

def sumbit():
    pred=predict()
    prediction=[1 if pred[i][0]>0.5 else 0 for i in range(pred.shape[0])]
    reply_test['prediction']=prediction
    output=reply_test[['query_id','reply_id','prediction']]
    output.to_csv('./tmp/prediction_query_reply.tsv',sep='\t',header=False,index=False)


if __name__=='__main__':
    submit()
