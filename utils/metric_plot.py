#!/usr/bin/env python
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style('whitegrid')

def plot_precision_recall_f1(epochs,val_precision,val_recall,save_path):
    epochs_=[i for i in range(1,epochs+1)]
    assert len(val_precision)==len(val_recall)
    f1=[2*val_precision[i]*val_recall[i]/(val_precision[i]+val_recall[i]) for i in range(len(val_precision))]
    
    plt.plot(epochs_,val_precision,'r',label='Precision')
    plt.plot(epochs_,val_recall,'y',label='Recall')
    plt.plot(epochs_,f1,'b',label='F1')
    plt.title('Model Performance on Validation Set')
    plt.xlabel('Epochs')
    plt.ylabel('P/R/F1')
    plt.legend()
    plt.savefig(save_path)
    
    best_f1_epoch=np.argmax(np.array(f1,dtype=float))+1
    
    return best_f1_epoch