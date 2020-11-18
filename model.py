import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset as BaseDataset


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


class RNet(nn.Module):

    def __init__(self, gru_in, gru_out):
        super(RNet, self).__init__()
        self.gru = nn.GRU(input_size=gru_in, hidden_size=gru_out, batch_first=True, bidirectional=True)
        self.M = nn.Parameter(torch.randn(2 * gru_out, 2 * gru_out))

    def forward(self, user_reviews, item_reviews):
        gru_u, hn = self.gru(user_reviews)
        gru_i, hn = self.gru(item_reviews)  # shape(batch, review_count * review_length, 2*gru_out)
        A = gru_i @ self.M.expand(gru_i.shape[0], -1, -1) @ gru_u.transpose(-1, -2)
        soft_u = torch.softmax(torch.max(A, dim=-2).values, dim=-1)  # column
        soft_i = torch.softmax(torch.max(A, dim=-1).values, dim=-1)  # row. shape(batch, review_count * review_length)
        atte_u = gru_u.transpose(-1, -2) @ soft_u.unsqueeze(-1)
        atte_i = gru_i.transpose(-1, -2) @ soft_i.unsqueeze(-1)  # shape(batch, 2*gru_out, 1)
        return gru_u.contiguous(), gru_i.contiguous(), soft_u, soft_i, atte_u.squeeze(-1), atte_i.squeeze(-1)


class SNet(nn.Module):

    def __init__(self, review_count, review_length, self_att_h, repr_size):
        super(SNet, self).__init__()
        self.r_count = review_count  # number of reviews for per user
        self.r_length = review_length  # ni in the paper
        self.self_att_h = self_att_h  # self-attention hidden size. It's us in the paper.
        self.repr_size = repr_size  # word embedding size. It's 2u in the paper

        self.Ms = nn.Parameter(torch.randn(self_att_h, repr_size))
        self.Ws = nn.Parameter(torch.randn(1, self_att_h))

    def forward(self, reviews, gru_hidden, word_soft):
        # self-attention for sentence-level sentiment.
        batch_size = reviews.shape[0]
        temp_batch_size = batch_size * self.r_count
        gru_hidden = gru_hidden.view(temp_batch_size, self.r_length, self.repr_size)
        Ms = self.Ms.expand(temp_batch_size, -1, -1)
        Ws = self.Ws.expand(temp_batch_size, -1, -1)
        sent_soft = torch.softmax(Ws @ torch.tanh(Ms @ gru_hidden.transpose(-1, -2)), dim=-1)  # (temp_batch,1,r_length)
        self_atte = gru_hidden.transpose(-1, -2) @ sent_soft.transpose(-1, -2)  # out(temp_batch, repr_size, 1)

        sentiment_emb = word_soft.view(temp_batch_size, self.r_length).sum(dim=-1, keepdim=True) * self_atte.squeeze(-1)
        sentiment_emb = sentiment_emb.view(batch_size, self.r_count, self.repr_size).sum(dim=-2)
        return sentiment_emb  # output(batch, repr_size)


class ReviewNet(nn.Module):

    def __init__(self, config, word_emb):
        super(ReviewNet, self).__init__()
        self.r_count = config.review_count
        self.r_length = config.review_length
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))

        self.r_net = RNet(self.embedding.embedding_dim, config.gru_hidden_size)  # Note: using Bi-GRU
        self.s_net_u = SNet(self.r_count, self.r_length, config.self_attention_hidden_size, config.gru_hidden_size * 2)
        self.s_net_i = SNet(self.r_count, self.r_length, config.self_attention_hidden_size, config.gru_hidden_size * 2)

        self.linear_u = nn.Linear(config.gru_hidden_size * 4, config.gru_hidden_size * 2, bias=False)
        self.linear_i = nn.Linear(config.gru_hidden_size * 4, config.gru_hidden_size * 2, bias=False)

    def forward(self, user_reviews, item_reviews):
        user_reviews = user_reviews.view(-1, self.r_count * self.r_length)
        item_reviews = item_reviews.view(-1, self.r_count * self.r_length)
        user_emb = self.embedding(user_reviews)
        item_emb = self.embedding(item_reviews)

        gru_u, gru_i, soft_u, soft_i, atte_u, atte_i = self.r_net(user_emb, item_emb)
        sentiment_u = self.s_net_u(user_emb, gru_u, soft_u)
        sentiment_i = self.s_net_i(item_emb, gru_i, soft_i)

        # Textual Matching
        repr_u = torch.cat([atte_u, sentiment_u], dim=-1)
        repr_i = torch.cat([atte_i, sentiment_i], dim=-1)
        represent = torch.tanh(self.linear_u(repr_u) + self.linear_i(repr_i))
        return represent  # output shape(batch, 2u) where u is GRU hidden size


class UMPR(nn.Module):

    def __init__(self, config, word_emb):
        super(UMPR, self).__init__()
        self.review_net = ReviewNet(config, word_emb)

        self.linear_fusion = nn.Linear(config.gru_hidden_size * 2, 1)

    def forward(self, user_reviews, item_reviews, reviews):
        review_net_represent = self.review_net(user_reviews, item_reviews)

        prediction = nn.functional.relu(self.linear_fusion(review_net_represent))
        return prediction
