import os
import sys
import time
import logging
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


def predict_mse(model, dataloader):
    mse, sample_count = 0, 0
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            process_bar(i + 1, len(dataloader), prefix='Evaluate')
            pred, loss = model(*batch)
            mse += F.mse_loss(pred, batch[-1].to(pred.device), reduction='sum').item()
            sample_count += len(pred)
    return mse / sample_count


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, photo_json, word_dict, config):
        self.word_dict = word_dict
        self.s_count = config.sent_count
        self.lowest_s_count = config.lowest_sent_count
        self.ui_s_count = config.ui_sent_count
        self.s_length = config.sent_length
        self.PAD_WORD_idx = word_dict[config.PAD_WORD]
        self.photo_count = config.photo_count

        df = pd.read_csv(data_path)
        df = df[df['review'].apply(lambda x: len(str(x))) > 5]
        df['numbered_r'], self.sent_pool = self._get_sentence_pool(df)
        self.retain_idx = [True] * df.shape[0]
        photos_name = self._get_photos_name(photo_json, df['itemID'], config.view_size)
        user_reviews = self._get_reviews(df)  # Gather reviews for every user without target review(i.e. u for i).
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

    def _get_sentence_pool(self, df):
        rev_sent_id = list()
        sent_pool = [[self.PAD_WORD_idx]]  # fill in with a null sentence
        for i, review in enumerate(df['review']):
            process_bar(i + 1, len(df), prefix=f'Loading sentences pool')
            sentences = [sent for sent in str(review).strip('. ').split('.') if len(sent) > 5]  # split review by "."
            rev_sent_id.append(list(range(len(sent_pool), len(sent_pool) + len(sentences))))
            for sent in sentences:
                sent_nums = [self.word_dict.get(w, self.PAD_WORD_idx) for w in sent.split()[:self.s_length]]  # cut sent
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

    def _get_photos_name(self, photos_json, item_id_list, view_size):
        photo_df = pd.read_json(photos_json, orient='records', lines=True)
        if 'label' not in photo_df.columns:
            photo_df['label'] = 'unknown'  # Due to amazon have no label.
        self.labels = photo_df['label'].drop_duplicates().to_list()
        assert len(self.labels) == view_size, f'By "{photos_json}", Config().view_size must be {len(self.labels)}!'

        photos_by_item = dict(list(photo_df[['photo_id', 'label']].groupby(photo_df['business_id'])))  # iid: df
        photos_name = []
        for idx, iid in enumerate(item_id_list):
            process_bar(idx + 1, len(item_id_list), prefix=f'Loading photos')
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
                pids = pids[:self.photo_count] + ['unk_name'] * (self.photo_count - len(pids))
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


def get_image(name, photo_dir, resize=(224, 224)):
    path = os.path.join(photo_dir, name + '.jpg')
    try:
        image = Image.open(path).convert('RGB').resize(resize)
        image = numpy.asarray(image) / 255
        return image.transpose((2, 0, 1))
    except Exception:
        return numpy.zeros([3] + list(resize))  # default


def batch_loader(batch_list, photo_dir, photo_size=(224, 224), pad=0):
    data = [list() for i in batch_list[0]]
    for sample in batch_list:
        for i, val in enumerate(sample):
            if i in (0, 1, 2):  # reviews val=[sent_id1, sent_id2, ...]
                data[i].append(val)
            if i == 3:  # photos
                data[i].append([[get_image(name, photo_dir, photo_size) for name in ps] for ps in val])
            if i == 4:  # ratings
                data[i].append(val)

    # pad sentences Ru and Ri
    max_count, max_len = 0, 0
    for ru, ri in zip(data[0], data[1]):
        max_count = max(max_count, max(len(ru), len(ri)))
        max_len = max(max_len, max(max([len(i) for i in ru]), max([len(i) for i in ri])))
    lengths = [0, 0, 0]
    data[0], lengths[0] = pad_reviews(data[0], max_count, max_len)
    data[1], lengths[1] = pad_reviews(data[1], max_count, max_len)
    data[2], lengths[2] = pad_reviews(data[2])

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
