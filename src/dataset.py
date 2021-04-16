import os
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy
import pandas as pd
import torch

from src.helpers import process_bar


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, photo_json, photo_dir, word2vec, config):
        self.s_count = config.sent_count
        self.lowest_s_count = config.lowest_sent_count
        self.ui_s_count = config.ui_sent_count
        self.s_length = config.sent_length
        self.photo_count = config.photo_count

        df = pd.read_csv(data_path)
        df = df[df['review'].apply(lambda x: len(str(x))) > 5]
        self.retain_idx = [True] * df.shape[0]
        photos_name = self._get_photos_name(photo_json, photo_dir, df['itemID'], config.view_size)
        df['numbered_r'], self.sent_pool = self._get_sentence_pool(df, word2vec)
        user_reviews = self._get_reviews(df)  # Gather reviews for every user without target review(i.e. u to i).
        item_reviews = self._get_reviews(df, 'item_num', 'user_num')
        ui_reviews = self._get_ui_review(df)

        self.data = (
            [v for i, v in enumerate(user_reviews) if self.retain_idx[i]],
            [v for i, v in enumerate(item_reviews) if self.retain_idx[i]],
            [v for i, v in enumerate(ui_reviews) if self.retain_idx[i]],
            [v for i, v in enumerate(photos_name) if self.retain_idx[i]],
            [v for i, v in enumerate(df['rating']) if self.retain_idx[i]],
        )

    def __getitem__(self, idx):
        return tuple(x[idx] for x in self.data)

    def __len__(self):
        return len(self.data[0])

    def _get_sentence_pool(self, df, w2v):
        rev_sent_id = list()
        sent_pool = [[0]]  # fill in with a null sentence
        for i, review in enumerate(df['review']):
            process_bar(i + 1, len(df), prefix=f'Loading sentences pool')
            sentences = [sent for sent in str(review).strip('. ').split('.') if len(sent) > 5]  # split review by "."
            rev_sent_id.append(list(range(len(sent_pool), len(sent_pool) + len(sentences))))
            for sent in sentences:
                sent_nums = w2v.sent2indices(sent)[:self.s_length]  # cut sent
                sent_pool.append(sent_nums)
        return rev_sent_id, sent_pool

    def _get_reviews(self, df, lead='user_num', costar='item_num'):
        reviews_by_lead = dict(list(df[[costar, 'numbered_r']].groupby(df[lead])))  # Information for every user/item
        results = []
        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            process_bar(idx + 1, len(df[lead]), prefix=f'Loading sentences group by {lead}')
            if not self.retain_idx[idx]:
                results.append(None)
                continue
            df_data = reviews_by_lead[lead_id]  # get information of lead, return DataFrame.
            reviews = df_data['numbered_r'][df_data[costar] != costar_id]  # get reviews without that u to i.
            sentences = [self.sent_pool[sent_id] for r in reviews for sent_id in r]
            if len(sentences) < self.lowest_s_count:
                self.retain_idx[idx] = False
                results.append(None)
                continue
            sentences.sort(key=lambda x: -len(x))  # sort by length of sentence.
            sentences = sentences[:self.s_count]
            results.append(sentences)
        return results  # shape(sample_count,sent_count)

    def _get_ui_review(self, df):
        reviews = list()
        for i, sentences in enumerate(df['numbered_r']):
            process_bar(i + 1, len(df), prefix=f'Loading ui sentences')
            if not self.retain_idx[i]:
                reviews.append(None)
                continue
            sentences = [self.sent_pool[i] for i in sentences]
            sentences.sort(key=lambda x: -len(x))  # sort by length of sentence.
            sentences = sentences[:self.ui_s_count]
            reviews.append(sentences)
        return reviews

    def _get_photos_name(self, photos_json, photo_dir, item_id_list, view_size):
        photo_df = pd.read_json(photos_json, orient='records', lines=True)
        if 'yelp' in photo_dir:
            self.labels = ['food', 'inside', 'outside', 'drink', 'menu']
        else:  # amazon
            self.labels = ['unknown']
            photo_df['label'] = self.labels[0]  # Due to amazon have no label.
        assert len(self.labels) == view_size, f'By "{photos_json}", Config().view_size must be {len(self.labels)}!'

        photos_by_item = dict(list(photo_df[['photo_id', 'label']].groupby(photo_df['business_id'])))  # iid: df
        photos_name = []
        for idx, iid in enumerate(item_id_list):
            process_bar(idx + 1, len(item_id_list), prefix=f'Loading photos\' path')
            if not self.retain_idx[idx]:
                photos_name.append(None)
                continue
            item_df = photos_by_item.get(iid, pd.DataFrame(columns=['photo_id', 'label']))  # all photos of this item.
            pid_by_label = dict(list(item_df['photo_id'].groupby(item_df['label'])))
            item_photos = list()
            for label in self.labels:
                pids = pid_by_label.get(label, pd.Series(dtype=str)).to_list()
                if len(pids) < 1:
                    self.retain_idx[idx] = False
                    item_photos = None
                    break
                pids = pids[:self.photo_count] + ['unknown'] * (self.photo_count - len(pids))
                pids = [os.path.join(photo_dir, name + '.jpg') for name in pids]
                item_photos.append(pids)
            photos_name.append(item_photos)
        return photos_name  # shape(sample_count,view_count,photo_count)


def pad_reviews(reviews, max_count=None, max_len=None, pad=0):
    if max_count is None:
        max_count = max(len(i) for i in reviews)
    reviews = [sents + [list()] * (max_count - len(sents)) for sents in reviews]

    lengths = [[max(1, len(sent)) for sent in sents] for sents in reviews]  # sentence length
    if max_len is None:
        max_len = max(max(i) for i in lengths)
    result = [[sent + [pad] * (max_len - len(sent)) for sent in sents] for sents in reviews]
    return result, lengths


def get_image(path, resize):
    try:
        image = cv2.imread(path)
        image = cv2.resize(image, resize)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        image = image / 255.0
        return image
    except Exception:
        return numpy.zeros([3] + list(resize))  # default


def batch_loader(batch_list, load_photo=True, photo_size=(224, 224), pad=0):
    # load all of photos using thread pool.
    photo_paths = [path for sample in batch_list for view in sample[3] for path in view]
    pool = ThreadPoolExecutor(max_workers=64)
    results = pool.map(lambda x: get_image(x, photo_size), photo_paths)

    data = [list() for i in batch_list[0]]
    for sample in batch_list:
        for i, val in enumerate(sample):
            if i in (0, 1, 2):  # reviews val=[sent_id1, sent_id2, ...]
                data[i].append(val)
            if i == 3 and load_photo:  # photos
                data[i].append([[next(results) for path in ps] for ps in val])
            if i == 4:  # ratings
                data[i].append(val)

    # pad sentences Ru and Ri
    max_count, max_len = 0, 0
    for ru, ri in zip(data[0], data[1]):
        max_count = max(max_count, max(len(ru), len(ri)))
        max_len = max(max_len, max(max([len(i) for i in ru]), max([len(i) for i in ri])))
    lengths = [0, 0, 0]
    data[0], lengths[0] = pad_reviews(data[0], max_count, max_len, pad=pad)
    data[1], lengths[1] = pad_reviews(data[1], max_count, max_len, pad=pad)
    data[2], lengths[2] = pad_reviews(data[2], pad=pad)

    return (
        torch.LongTensor(data[0]),
        torch.LongTensor(data[1]),
        torch.LongTensor(data[2]),
        torch.LongTensor(lengths[0]),
        torch.LongTensor(lengths[1]),
        torch.LongTensor(lengths[2]),
        torch.Tensor(data[3]),
        torch.Tensor(data[4]),
    )
