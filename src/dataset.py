import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy
import pandas as pd
import torch
from tqdm import tqdm, trange


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, photo_json, photo_dir, word2vec, config):
        self.max_s_count = config.max_sent_count
        self.min_s_count = config.min_sent_count
        self.max_ui_s_count = config.max_ui_sent_count
        self.max_s_length = config.max_sent_length
        self.photo_count = config.photo_count
        self.views = config.views

        df = pd.read_csv(data_path)
        df['review'] = df['review'].apply(lambda x: [
            s for s in [
                word2vec.sent2indices(sent)[:self.max_s_length] for sent in
                (str(x).strip('. ').split('.') if config.review_level == 'sentence' else [str(x)])
            ]
            if len(s) > 5  # sentence with length not more than 5 will be removed.
        ])  # Each review will be formatted to [[wid1, wid2,...],[wid1,...],...] under the sentence level.

        self.retain_idx = [len(x) > 0 for x in df['review']]  # review with none of sentence will be removed.

        photos_name = self._get_photos_name(photo_json, photo_dir, df['itemID'])
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

    def _get_reviews(self, df, lead='user_num', costar='item_num'):
        groups = defaultdict(list)
        for lead_id, costar_id, review in tqdm(zip(df[lead], df[costar], df['review']), total=len(df), desc='Grouping'):
            groups[lead_id].append([costar_id, review])

        results = []
        with trange(len(df[lead]), desc=f'Loading sentences of {lead}') as t_bar:
            for i, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
                t_bar.update()
                if not self.retain_idx[i]:
                    results.append(None)
                    continue

                reviews = [r for cid, r in groups[lead_id] if cid != costar_id]  # get reviews without that u to i.
                sentences = [sent for review in reviews for sent in review]
                if len(sentences) < self.min_s_count:
                    self.retain_idx[i] = False
                    results.append(None)
                    continue
                if len(sentences) > self.max_s_count:
                    sentences.sort(key=lambda x: -len(x))  # sort by length of sentence.
                    sentences = sentences[:self.max_s_count]
                results.append(sentences)
        return results  # shape(sample_count,sent_count)

    def _get_ui_review(self, df):
        reviews = list()
        for i, sentences in tqdm(enumerate(df['review']), desc='Loading ui sentences', total=len(df['review'])):
            if not self.retain_idx[i]:
                reviews.append(None)
                continue
            if len(sentences) > self.max_ui_s_count:
                sentences.sort(key=lambda x: -len(x))  # sort by length of sentence.
                sentences = sentences[:self.max_ui_s_count]
            reviews.append(sentences)
        return reviews

    def _get_photos_name(self, photos_json, photo_dir, item_id_list):
        photo_df = pd.read_json(photos_json, orient='records', lines=True)
        if 'label' not in photo_df.columns:
            photo_df['label'] = self.views[0]  # Because amazon have no label.

        photo_groups = defaultdict(dict)
        for row in tqdm(photo_df.itertuples(), desc='Reading photos\' DataFrame', total=len(photo_df)):
            bid = getattr(row, 'business_id')
            pid = getattr(row, 'photo_id')
            label = getattr(row, 'label')
            if label in self.views:
                if label not in photo_groups[bid]:
                    photo_groups[bid][label] = list()
                photo_groups[bid][label].append(pid)

        photos_paths = []
        for idx, bid in tqdm(enumerate(item_id_list), desc='Loading photos\' path', total=len(item_id_list)):
            if not self.retain_idx[idx]:
                photos_paths.append(None)
                continue
            item_photos = list()
            for label in self.views:  # Each view
                pids = photo_groups[bid].get(label, list())
                if len(pids) < 1:  # Too few photos of this view
                    self.retain_idx[idx] = False
                    item_photos = None
                    break
                pids = [os.path.join(photo_dir, pids[j] + '.jpg') for j in range(0, min(len(pids), self.photo_count))]
                if len(pids) < self.photo_count:  # Insufficient length
                    pids.extend(['unknown'] * (self.photo_count - len(pids)))
                item_photos.append(pids)
            photos_paths.append(item_photos)
        return photos_paths  # shape(sample_count,view_count,photo_count)


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


def batch_loader(batch_list, ignore_photos=False, photo_size=(224, 224), pad=0):
    # load all of photos using thread pool.
    photo_paths = [path for sample in batch_list for view in sample[3] for path in view]
    pool = ThreadPoolExecutor()
    results = pool.map(lambda x: get_image(x, photo_size), photo_paths)
    pool.shutdown()

    data = [list() for i in batch_list[0]]
    for sample in batch_list:
        for i, val in enumerate(sample):
            if i in (0, 1, 2):  # reviews val=[sent_id1, sent_id2, ...]
                data[i].append(val)
            if not ignore_photos and i == 3:  # photos
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
