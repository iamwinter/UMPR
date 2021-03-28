UMPR
===
Implementation for the paper：  
>Xu, Cai, Ziyu Guan, Wei Zhao, Quanzhou Wu, Meng Yan, Long Chen, and Qiguang Miao.
 "Recommendation by Users' Multi-modal Preferences for Smart City Applications."
 IEEE Transactions on Industrial Informatics (2020).

# Environments
+ python 3.8
+ pytorch 1.7

# Dataset and Word Embedding

```
UMPR
│
├─data
│  │
│  ├─amazonCSJ
│  │       reviews_Clothing_Shoes_and_Jewelry.json.gz
│  │       meta_Clothing_Shoes_and_Jewelry.json.gz
│  │
│  ├─music
│  │       reviews_Digital_Music.json.gz
│  │       meta_Digital_Music.json.gz
│  │
│  └─yelp
│     │    yelp_academic_dataset_review.json
│     │    photos.json
│     │
│     └─photos
│              *.jpg
│
└─embedding
           glove.6B.50d.txt
           punctuations.txt
           stopwords.txt
```

+ Dataset Amazon(2014) http://jmcauley.ucsd.edu/data/amazon/links.html
+ Dataset Yelp(2020) https://www.yelp.com/dataset
+ Word Embedding https://nlp.stanford.edu/projects/glove

# Running

+ Firstly, execute `data_process.py` to generate 
`train.csv`,`valid.csv`,`test.csv`,`photos.json`.
```shell script
python data/data_process.py --data_type amazon \
    --data_path ./data/music/reviews_Digital_Music.json.gz \
    --meta_path ./data/music/meta_Digital_Music.json.gz \
    --save_dir ./data/music \
    --train_rate 0.8
```

+ For amazon(**not yelp**), execute `down_photos.py` to download `photos/*.jpg`.
```shell script
python data/down_photos.py --photos_json ./data/music/photos.json
```

+ Train and evaluate the model. if use yelp dataset, you must set view_size=5.
```shell script
python main.py --data_dir ./data/music --view_size 1
```
+ Test only:
```shell script
python test.py --data_dir ./data/music --view_size 1 --model_path ./model/default.pt
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
        <td>0.925538</td>
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


# Experiment Notebook

+ 2021.03.22

  - 发现`torchvision.models.models.vgg16`最后一层是`Linear`，输出值很大（~1e6）。
  - 公式18涉及除法，有可能造成除以0的情况。解决：分母额外加1e-6。

+ 2021.03.24

  - 数据集处理技巧：评论的句子按长度排序，长句优先用于训练，短句则大概率在被对齐句数时丢弃。
  - GRU改为不定长输入。原方法是输入等长序列。

+ 2021.03.26

  - 对于图片集，一次性读入因内存不足而退出。解决：每个batch训练/测试前即时从磁盘中读取图片。

+ 2021.03.28

  - 尝试yelp数据集时，内存还是不足，只好把sentence语句保存到“语句池”，每个batch训练时再读取sentence
