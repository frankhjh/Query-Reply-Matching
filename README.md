# Use Bert to solve Reply-Query-Matching Task

## Basic Introduction
The company who hold this competition is *Shell House-Searching*, one of the most famous Internet real estate company in China. In order to provide better reply service for online queries from large volume of clients, it requires a AI model for doing this, that is, given the real query data, it requires us to build a model to find the best/realistic reply.

## Data
The form of data it provides is text,including the query and reply. For each query, it provides several replies, one of them is real reply, while the others are all fake. The **query-real reply** pair is labeled as **1** and the **query-fake reply** pair is labeled as **0**. our aim is to train a classification model to identify them.

## Model
My first idea is to use CNN and LSTM, but unfortunately,both of them have poor performance on the test set.I also used the stacking method to combine them, but the performance improvement is also not significant. Then I tried the famous pre-trained model:**Bert**. as I expected, the performance of it on this task is really good.

The final F1 score on test set is **0.76341018**, the rank is top **15%**

## Performance on Validation set
Below shows the Precision/Recall/F1 score of model on validation set in each epoch during training.

![Precision/Recall/F1 score](https://github.com/frankhjh/Reply-Query-Matching/blob/main/tmp/metric_scores.png  'Model Performance on Validation set')





## Try yourself
