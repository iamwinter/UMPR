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
  1. dataset(`/data/music/Digital_Music_5.json.gz`)  
   Download from http://jmcauley.ucsd.edu/data/amazon (Choose Digital Music)

# Running

Preprocess origin dataset in json format to train.csv,valid.csv and test.csv.  
**Rewrite some necessary settings** in this file before running it. 
```
python preprocess.py
```

Train and evaluate the model:
```
python main.py
```
