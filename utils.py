import time
import pandas as pd
import torch
from torch.nn import functional as F


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


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


def predict_mse(model, dataloader):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_reviews, item_reviews, reviews, ratings = map(lambda x: x.to(next(model.parameters()).device), batch)
            predict = model(user_reviews, item_reviews, reviews)
            mse += F.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count


def split_list(L, key):
    """
    split a list by key
    :param L: list[]
    :param key:
    :return: 2-dim list
    """
    result = []
    temp = []
    for x in L:
        if x == key:
            if len(temp) > 0:
                result.append(temp)
            temp = []
        else:
            temp.append(x)
    return result


def pad_list(L, dim1, dim2, pad_elem=0):
    """
    二维list调整长宽，截长补短
    :param L:
    :param dim1:
    :param dim2:
    :param pad_elem:
    :return: 二维list
    """
    L = L[:dim1] + [[pad_elem] * dim2] * (dim1 - len(L))  # dim 1
    L = [r[:dim2] + [pad_elem] * (dim2 - len(r)) for r in L]  # dim 2
    return L


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, word_dict, config):
        self.word_dict = word_dict
        self.s_count = config.sent_count
        self.ui_s_count = config.ui_sent_count
        self.s_length = config.sent_length
        self.lowest_s_count = config.lowest_sent_count  # lowest amount of sentences wrote by exactly one user/item
        self.PAD_WORD_idx = word_dict[config.PAD_WORD]
        self.SPLIT_idx = word_dict['.']  # split to sentences

        df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating'])
        df['review'] = df['review'].apply(self._sent2id)
        self.delete_idx = set()  # Save the indices of empty samples, delete them at last.
        user_reviews = self._get_reviews(df)  # Gather reviews for every user without target review(i.e. u for i).
        item_reviews = self._get_reviews(df, 'itemID', 'userID')
        ui_reviews = self._get_rui(df['review'])
        retain_idx = [idx for idx in range(len(df)) if idx not in self.delete_idx]

        self.user_reviews = user_reviews[retain_idx]
        self.item_reviews = item_reviews[retain_idx]
        self.ui_reviews = ui_reviews[retain_idx]
        self.rating = torch.Tensor(df['rating'].to_list())[retain_idx]

    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.ui_reviews[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_reviews(self, df, lead='userID', costar='itemID'):
        # For every sample(user,item), gather reviews for user/item.
        reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))  # Information for every user/item
        ret_sentences = []
        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            df_data = reviews_by_lead[lead_id]  # get information of lead, return DataFrame.
            reviews = df_data['review'][df_data[costar] != costar_id].to_list()  # get reviews without review u for i.
            sentences = [sent for r in reviews for sent in split_list(r, self.SPLIT_idx)]  # sentence list
            if len(sentences) < self.lowest_s_count:
                self.delete_idx.add(idx)
            sentences = pad_list(sentences, self.s_count, self.s_length, self.PAD_WORD_idx)
            ret_sentences.append(sentences)
        return torch.LongTensor(ret_sentences)

    def _get_rui(self, reviews):
        ui_reviews = []
        for review in reviews:
            r = split_list(review, self.SPLIT_idx)
            r = pad_list(r, self.ui_s_count, self.s_length, self.PAD_WORD_idx)
            ui_reviews.append(r)
        return torch.LongTensor(ui_reviews)

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
