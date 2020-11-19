import time
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset as BaseDataset


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


def mse_loss(model, dataloader):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_reviews, item_reviews, reviews, ratings = map(lambda x: x.to(next(model.parameters()).device), batch)
            predict = model(user_reviews, item_reviews, reviews)
            mse += F.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count


class Dataset(BaseDataset):
    def __init__(self, data_path, word_dict, config):
        self.word_dict = word_dict
        self.r_count = config.review_count
        self.r_length = config.review_length
        self.lowest_r_count = config.lowest_review_count  # lowest amount of reviews wrote by exactly one user/item
        self.PAD_WORD_idx = word_dict[config.PAD_WORD]

        df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating'])
        df['review'] = df['review'].apply(self._review2id)
        self.delete_idx = set()  # Save the indices of empty samples, delete them at last.
        user_reviews = self._get_reviews(df)  # Gather reviews for every user without target review(i.e. u for i).
        item_reviews = self._get_reviews(df, 'itemID', 'userID')
        reviews = [self._adjust_review_list([x], 1, self.r_length) for x in df['review']]
        retain_idx = [idx for idx in range(len(df)) if idx not in self.delete_idx]

        self.user_reviews = user_reviews[retain_idx]
        self.item_reviews = item_reviews[retain_idx]
        self.reviews = torch.LongTensor(reviews).view(-1, self.r_length)[retain_idx]
        self.rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)[retain_idx]

    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.reviews[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_reviews(self, df, lead='userID', costar='itemID'):
        # For every sample(user,item), gather reviews for user/item.
        reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))  # Information for every user/item
        lead_reviews = []
        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            df_data = reviews_by_lead[lead_id]  # get information of lead, return DataFrame.
            reviews = df_data['review'][df_data[costar] != costar_id].to_list()  # get reviews without review u for i.
            if len(reviews) < self.lowest_r_count:
                self.delete_idx.add(idx)
            reviews = self._adjust_review_list(reviews, self.r_count, self.r_length)
            lead_reviews.append(reviews)
        return torch.LongTensor(lead_reviews)

    def _adjust_review_list(self, reviews, r_count, r_length):
        reviews = reviews[:r_count] + [[self.PAD_WORD_idx] * r_length] * (r_count - len(reviews))  # Certain count.
        reviews = [r[:r_length] + [0] * (r_length - len(r)) for r in reviews]  # Certain length of review.
        return reviews

    def _review2id(self, review):  # Split a sentence into words, and map each word to a unique number by dict.
        if not isinstance(review, str):
            return []
        wids = []
        for word in review.split():
            if word in self.word_dict:
                wids.append(self.word_dict[word])  # word to unique number by dict.
            else:
                wids.append(self.PAD_WORD_idx)
        return wids
