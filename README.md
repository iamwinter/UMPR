UMPR
===
Implementation for the paper：  
>Xu, Cai, Ziyu Guan, Wei Zhao, Quanzhou Wu, Meng Yan, Long Chen, and Qiguang Miao.
 "Recommendation by Users' Multi-modal Preferences for Smart City Applications."
 IEEE Transactions on Industrial Informatics (2020).

ToDo: C-Net公用；Visual-Net。

# Environments
  + python 3.8
  + pytorch 1.7

# Dataset

1. Amazon(2014) http://jmcauley.ucsd.edu/data/amazon/links.html
2. Yelp(2020) https://www.yelp.com/dataset

For example: `data/reviews_Clothing_Shoes_and_Jewelry_5.json.gz`.
Then you should execute following command to create train/validation/test dataset.
```shell script
python data_preprocess.py --data_path Digital_Music_5.json --data_source amazon --train_rate 0.8 --save_dir ./music
```

# Word Embedding

+ Download from https://nlp.stanford.edu/projects/glove

For example:`embedding/glove.6B.50d.txt`

# Running

Train and evaluate the model:
```
python main.py
```

# Experiment

<p align="center" style="margin: 0">
Table 1. 
Performance comparison (mean squared error) on several datasets.
</p>
<table align="center">
    <tr>
        <th>Dataset(number of reviews)</th>
        <th>MF</th>
        <th>NeulMF</th>
        <th>DeepCoNN</th>
        <th>TransNets</th>
        <th>MPCN</th>
        <th>UMPR-R</th>
        <th>UMPR</th>
    </tr>
    <tr>
        <td>Amazon Music small (64,706)</td>
        <td>0.900899</td>
        <td>0.822472</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>1.117017</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Amazon Music (836,006)</td>
        <td>0.875224</td>
        <td>0.825261</td>
    </tr>
    <tr>
        <td>Amazon Clothing, Shoes and Jewelry (5,748,920)</td>
        <td>1.512551</td>
        <td>1.502135</td>
    </tr>
    <tr>
        <td>Yelp (8,021,121)</td>
        <td>2.171064</td>
        <td>2.041674</td>
    </tr>
</table>

**MF**: General Matrix Factorization.
[Details](https://github.com/iamwinter/MatrixFactorization)

**NeuMF**: Neural Collaborative Filtering.
[Details](https://github.com/iamwinter/NeuralCollaborativeFiltering)

**DeepCoNN**: [Details](https://github.com/iamwinter/DeepCoNN)

**UMPR-R**: only review network part of UMPR.

**UMPR**: Our complete model.
