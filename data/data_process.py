import argparse
import gzip
import json
import os
import sys
import time
import uuid

import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import WordPunctTokenizer


def process_dataset(reviews_path, meta_path, save_dir, train_rate, select_cols):
    os.makedirs(save_dir, exist_ok=True)

    print(f'#### Read {reviews_path}')
    if reviews_path.endswith('.gz'):
        f = gzip.open(reviews_path, 'rb')
    else:
        f = open(reviews_path, 'r', encoding='UTF-8')
    data = []
    for line in f.readlines():
        item = json.dumps(eval(line))
        item = json.loads(item)
        data.append([item[i] for i in select_cols])
    f.close()
    df = pd.DataFrame(data, columns=['userID', 'itemID', 'review', 'rating'])  # Rename above columns for convenience

    # map user(or item) to number
    df['user_num'] = df.groupby(df['userID']).ngroup()
    df['item_num'] = df.groupby(df['itemID']).ngroup()

    with open(os.path.join(sys.path[0], '../embedding/stopwords.txt')) as f:  # stop words
        stop_words = set(f.read().splitlines())
    with open(os.path.join(sys.path[0], '../embedding/punctuations.txt')) as f:  # useless punctuations
        punctuations = set(f.read().splitlines())
        punctuations.remove('.')

    def clean_review(review):  # clean a review using stop words and useless punctuations
        review = review.lower()
        for p in punctuations:
            review = review.replace(p, ' ')  # replace punctuations by space
        review = WordPunctTokenizer().tokenize(review)  # split words
        review = [word for word in review if word not in stop_words]  # remove stop words
        # review = [nltk.WordNetLemmatizer().lemmatize(word) for word in review]  # extract root of word
        return ' '.join(review)

    df = df.drop(df[[not isinstance(x, str) or len(x) == 0 for x in df['review']]].index)  # erase null reviews
    df['review'] = df['review'].apply(clean_review)

    train, valid = train_test_split(df, test_size=1 - train_rate, random_state=3)  # split dataset including random
    valid, test = train_test_split(valid, test_size=0.5, random_state=4)
    train.to_csv(os.path.join(save_dir, 'train.csv'), index=False)
    valid.to_csv(os.path.join(save_dir, 'valid.csv'), index=False)
    test.to_csv(os.path.join(save_dir, 'test.csv'), index=False)
    print(f'#### Saved dataset({len(df)} reviews, {len(df["user_num"].drop_duplicates())} users, '
          f'{len(df["item_num"].drop_duplicates())} items): '
          f'train.csv({len(train)}), valid.csv({len(valid)}), test.csv({len(test)})')

    # users_df = df[['userID']].drop_duplicates()
    # items_df = df[['itemID']].drop_duplicates()
    # users_df.to_csv(os.path.join(save_dir, 'users.csv'), index=False)
    # items_df.to_csv(os.path.join(save_dir, 'items.csv'), index=False)
    # print(f'#### Saved users.csv({len(users_df)}), items.csv({len(items_df)}).')

    try:
        print(f'#### Read {meta_path}')
        if meta_path.endswith('.gz'):
            f = gzip.open(meta_path, 'rb')
        else:
            f = open(meta_path, 'r', encoding='UTF-8')
        photos = []
        items_set = set(df['itemID'])
        for line in f.readlines():
            item = json.dumps(eval(line))
            item = json.loads(item)
            if 'imUrl' in item and item['asin'] in items_set:  # Only download pictures of existing items.
                photo_name = uuid.uuid4().hex[:16]
                photos.append([item['asin'], photo_name, item['imUrl']])
        f.close()
        df = pd.DataFrame(photos, columns=['business_id', 'photo_id', 'imUrl'])
        df.to_json(os.path.join(save_dir, 'photos.json'), orient='records', lines=True)
        print(f'#### Saved photos.json({len(photos)} pictures).')
    except Exception:
        print(f'#### Failed to read {meta_path} or its content is damaged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', dest='data_type', default='amazon')
    parser.add_argument('--data_path', dest='data_path', default=os.path.join(sys.path[0], 'music/reviews_Digital_Music.json.gz'))
    parser.add_argument('--meta_path', dest='meta_path', default=os.path.join(sys.path[0], 'music/meta_Digital_Music.json.gz'))
    parser.add_argument('--save_dir', dest='save_dir', default=os.path.join(sys.path[0], 'music'))
    parser.add_argument('--train_rate', dest='train_rate', default=0.8)
    args = parser.parse_args()

    col_name = ['reviewerID', 'asin', 'reviewText', 'overall']  # default amazon
    if args.data_type == 'yelp':
        col_name = ['user_id', 'business_id', 'text', 'stars']
    os.makedirs(args.save_dir, exist_ok=True)

    start_time = time.perf_counter()
    process_dataset(args.data_path, args.meta_path, args.save_dir, float(args.train_rate), col_name)
    end_time = time.perf_counter()
    print(f'## preprocess.py: Data loading complete! Time used {end_time - start_time:.0f} seconds.')