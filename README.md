UMPR
===
>Implementation using pytorch for the paper：  
Xu, Cai, Ziyu Guan, Wei Zhao, Quanzhou Wu, Meng Yan, Long Chen, and Qiguang Miao. "Recommendation by Users' Multi-modal Preferences for Smart City Applications." IEEE Transactions on Industrial Informatics (2020).

Only Review Network has been implemented.

# Environments
  + python 3.8
  + pytorch 1.7

# Dataset
  You need to prepare the following documents:  
  1. dataset(`data/amazonCSJ/reviews_Clothing_Shoes_and_Jewelry_5.json.gz`)  
    It contains 278677 reviews
    and download from http://jmcauley.ucsd.edu/data/amazon/links.html

# Running

+ Preprocess origin dataset in json format to train.csv,valid.csv and test.csv.  
**Rewrite some necessary settings** in this file before running it. 
```
python preprocess.py
```
+ Train and evaluate the model:
```
python main.py --device cuda:0
```

>More arguments (You can also see them at `config.py`):  
--device  
--train_epochs  
--batch_size  
--learning_rate  
--l2_regularization  
--learning_rate_decay  
--review_count  
--review_length  
--lowest_review_count  
--PAD_WORD  
--gru_hidden_size  
--self_attention_hidden_size  
