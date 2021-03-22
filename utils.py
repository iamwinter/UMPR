import os
import sys
import time
import logging
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor

import numpy
from PIL import Image
import pandas as pd
import torch
from torch.nn import functional as F


def get_logger(log_file=None, file_level=logging.INFO, stdout_level=logging.DEBUG, logger_name=__name__):
    logging.root.setLevel(0)
    formatter = logging.Formatter('%(asctime)s %(levelname)5s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    _logger = logging.getLogger(logger_name)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level=file_level)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=stdout_level)
    stream_handler.setFormatter(formatter)
    _logger.addHandler(stream_handler)
    return _logger


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


def process_bar(current, total, prefix='', auto_rm=True):
    bar = '=' * int(current / total * 50)
    bar = f' {prefix} |{bar.ljust(50)}| ({current}/{total}) {current / total:.1%} | '
    print(bar, end='\r', flush=True)
    if auto_rm and current == total:
        print(end=('\r' + ' ' * len(bar) + '\r'), flush=True)


def load_embedding(word2vec_file):
    with open(word2vec_file, encoding='utf-8') as f:
        word_emb = list()
        word_dict = dict()
        word_emb.append([0])
        word_dict['<UNK>'] = 0
        for line in f.readlines():
            tokens = line.split(' ')
            word_emb.append([float(i) for i in tokens[1:]])
            word_dict[tokens[0]] = len(word_dict)
        word_emb[0] = [0] * len(word_emb[1])
    return word_emb, word_dict


def load_photos(photos_dir, resize=(64, 64), max_workers=8):
    paths = []
    for name in os.listdir(photos_dir):
        path = os.path.join(photos_dir, name)
        if os.path.isfile(path) and name.endswith('jpg'):
            paths.append(path)

    def load_image(img_path):
        try:
            image = Image.open(img_path).convert('RGB').resize(resize)
            image = numpy.asarray(image) / 255
            return os.path.basename(img_path)[:-4], image.transpose((2, 0, 1))
        except Exception:
            return img_path, None  # Damaged picture: {img_path}

    pool = ThreadPoolExecutor(max_workers=max_workers)  # to read {len(paths)} pictures
    tasks = [pool.submit(load_image, path) for path in paths]

    photos_dict = dict()
    damaged = []
    for i, task in enumerate(as_completed(tasks)):
        name, photo = task.result()
        if photo is not None:
            photos_dict[name] = photo
        else:
            damaged.append(name)
        process_bar(i + 1, len(tasks), prefix='Loading photos')

    for name in damaged:
        print(f'## Failed to open {name}.jpg')
    return photos_dict


def predict_mse(model, dataloader):
    device = next(model.parameters()).device
    mse, sample_count = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            user_reviews, item_reviews, reviews, photos, ratings = map(lambda x: x.to(device), batch)
            predict, loss = model(user_reviews, item_reviews, reviews, photos, ratings)
            mse += F.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
            process_bar(i + 1, len(dataloader), prefix='Evaluate')
    return mse / sample_count


def split_list(arr, key):  # split a list by key
    result = []
    temp = []
    for x in arr:
        if x == key:
            if len(temp) > 0:
                result.append(temp)
            temp = []
        else:
            temp.append(x)
    return result


def pad_list(arr, dim1, dim2, pad_elem=0):  # 二维list调整长宽，截长补短
    arr = arr[:dim1] + [[pad_elem] * dim2] * (dim1 - len(arr))  # dim 1
    arr = [r[:dim2] + [pad_elem] * (dim2 - len(r)) for r in arr]  # dim 2
    return arr


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, photo_json, word_dict, config):
        self.word_dict = word_dict
        self.s_count = config.sent_count
        self.ui_s_count = config.ui_sent_count
        self.s_length = config.sent_length
        self.lowest_s_count = config.lowest_sent_count  # lowest amount of sentences wrote by exactly one user/item
        self.PAD_WORD_idx = word_dict[config.PAD_WORD]
        self.SPLIT_idx = word_dict['.']  # split to sentences

        df = pd.read_csv(data_path)
        df['review'] = df['review'].apply(self._sent2id)
        self.retain_idx = [True] * len(df)  # Save the indices of empty samples, delete them at last.
        user_reviews = self._get_reviews(df)  # Gather reviews for every user without target review(i.e. u for i).
        item_reviews = self._get_reviews(df, 'item_num', 'user_num')
        ui_reviews = self._get_rui(df['review'])
        photos_id = self._load_photos_id(photo_json, df['itemID'], config.view_size)

        self.user_reviews = user_reviews[self.retain_idx]
        self.item_reviews = item_reviews[self.retain_idx]
        self.ui_reviews = ui_reviews[self.retain_idx]
        self.photos_id = photos_id[self.retain_idx]
        self.rating = numpy.asarray(df['rating'])[self.retain_idx]

    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.ui_reviews[idx], \
               self.photos_id[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_reviews(self, df, lead='user_num', costar='item_num', max_workers=8):
        # For every sample(user,item), gather reviews for user/item.
        reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))  # Information for every user/item

        def gather_review(idx, lead_id, costar_id):
            df_data = reviews_by_lead[lead_id]  # get information of lead, return DataFrame.
            reviews = df_data['review'][df_data[costar] != costar_id]  # get reviews without review u for i.
            sentences = [sent for r in reviews for sent in split_list(r, self.SPLIT_idx)]  # sentence list
            if len(sentences) < self.lowest_s_count:
                self.retain_idx[idx] = False
            return idx, sentences

        pool = ThreadPoolExecutor(max_workers=max_workers)
        tasks = [pool.submit(gather_review, i, x[0], x[1]) for i, x in enumerate(zip(df[lead], df[costar]))]

        ret_sentences = [[]] * len(tasks)
        for i, task in enumerate(as_completed(tasks)):
            idx, sents = task.result()
            ret_sentences[idx] = sents
            process_bar(i + 1, len(tasks), prefix=f'Loading sentences of {lead}')
        return numpy.asarray(ret_sentences, dtype=object)

    def _get_rui(self, reviews):
        ui_reviews = []
        for review in reviews:
            r = split_list(review, self.SPLIT_idx)
            # r = pad_list(r, self.ui_s_count, self.s_length, self.PAD_WORD_idx)
            ui_reviews.append(r)
        return numpy.asarray(ui_reviews, dtype=object)

    def _sent2id(self, sent):  # Split a sentence into words, and map each word to a unique number by dict.
        if not isinstance(sent, str):
            return []
        wids = []
        for word in sent.split():
            if word in self.word_dict:
                wids.append(self.word_dict[word])  # word to unique number by dict.
            else:
                wids.append(self.PAD_WORD_idx)
        return wids

    def _load_photos_id(self, photos_json, item_id_list, view_size):
        photo_df = pd.read_json(photos_json, orient='records', lines=True)
        if 'label' not in photo_df.columns:
            photo_df['label'] = 'None'  # Due to amazon have no label.
        label_index = dict([(label, i) for i, label in enumerate(photo_df['label'].drop_duplicates())])  # label: index
        if len(label_index) != view_size:
            print(f'Number of labels in photos.json is {len(label_index)}! Set Config().view_size={len(label_index)}!')
            exit(1)

        user_photos = dict(list(photo_df[['photo_id', 'label']].groupby(photo_df['business_id'])))  # iid: df

        photos_id = []
        for idx, iid in enumerate(item_id_list):
            item_df = user_photos.get(iid, pd.DataFrame(columns=['photo_id', 'label']))  # all photos of this item.

            item_photos = [[]] * len(label_index)  # shape(label_size, photos count)
            for label, pids in dict(list(item_df['photo_id'].groupby(item_df['label']))).items():
                item_photos[label_index[label]] = pids.to_list()
                if len(pids) < 1:  # Too few photos of this view.
                    self.retain_idx[idx] = False

            photos_id.append(item_photos)
        return numpy.asarray(photos_id, dtype=object)  # shape(samples, label_size, photo_count)


def batch_loader(batch_list, config, photos_dict):
    photo_size = list(photos_dict.values())[0].shape
    u_sents, i_sents, ui_sents, photos, ratings = [], [], [], [], []
    for u_s, i_s, ui_s, photos_ids, rating in batch_list:
        u_sents.append(pad_list(u_s, config.sent_count, config.sent_length))
        i_sents.append(pad_list(i_s, config.sent_count, config.sent_length))
        ui_sents.append(pad_list(ui_s, config.ui_sent_count, config.sent_length))

        cur_photos = []  # photos of current item
        for pids_of_label in photos_ids:  # There are several photos of current item.
            label_photos = [photos_dict[pid] for pid in pids_of_label if pid in photos_dict]  # convert ids to photos
            if len(cur_photos) >= config.photo_count:
                label_photos = label_photos[:config.photo_count]  # for each view(label)
            while len(label_photos) < config.photo_count:
                label_photos.append(numpy.zeros(photo_size))  # shape(label_count, photo_count)
            cur_photos.append(label_photos)

        photos.append(cur_photos)  # shape(batch_size, view_size, photo_count, C,H,W)
        ratings.append(rating)
    return torch.LongTensor(u_sents), torch.LongTensor(i_sents), torch.LongTensor(ui_sents), \
           torch.Tensor(photos), torch.Tensor(ratings)
