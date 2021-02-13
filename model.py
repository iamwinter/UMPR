import torch
from torch import nn


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

    def __init__(self, sent_count, sent_length, self_att_h, repr_size):
        super(SNet, self).__init__()
        self.s_count = sent_count  # number of reviews for per user
        self.s_length = sent_length  # ni in the paper
        self.self_att_h = self_att_h  # self-attention hidden size. It's us in the paper.
        self.repr_size = repr_size  # word embedding size. It's 2u in the paper

        self.Ms = nn.Parameter(torch.randn(self_att_h, repr_size))
        self.Ws = nn.Parameter(torch.randn(1, self_att_h))

    def forward(self, reviews, gru_hidden, word_soft):
        # self-attention for sentence-level sentiment.
        batch_size = reviews.shape[0]
        temp_batch_size = batch_size * self.s_count
        gru_hidden = gru_hidden.view(temp_batch_size, self.s_length, self.repr_size)
        Ms = self.Ms.expand(temp_batch_size, -1, -1)
        Ws = self.Ws.expand(temp_batch_size, -1, -1)
        sent_soft = torch.softmax(Ws @ torch.tanh(Ms @ gru_hidden.transpose(-1, -2)), dim=-1)  # (temp_batch,1,r_length)
        self_atte = gru_hidden.transpose(-1, -2) @ sent_soft.transpose(-1, -2)  # out(temp_batch, repr_size, 1)

        sentiment_emb = word_soft.view(temp_batch_size, self.s_length).sum(dim=-1, keepdim=True) * self_atte.squeeze(-1)
        sentiment_emb = sentiment_emb.view(batch_size, self.s_count, self.repr_size).sum(dim=-2)
        return sentiment_emb  # output(batch, repr_size)


class CNet(nn.Module):

    def __init__(self, gru_in, gru_out):
        super(CNet, self).__init__()
        self.gru = nn.GRU(input_size=gru_in, hidden_size=gru_out, batch_first=True, bidirectional=True)

    def forward(self, user_reviews, item_reviews, ui_review):
        gru_u, hn = self.gru(user_reviews)
        gru_i, hn = self.gru(item_reviews)  # shape(batch, review_length, 2*gru_out)
        # todo


class ReviewNet(nn.Module):

    def __init__(self, config, word_emb):
        super(ReviewNet, self).__init__()
        self.s_count = config.sent_count
        self.s_length = config.sent_length
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))

        self.r_net = RNet(self.embedding.embedding_dim, config.gru_hidden_size)  # Note: using Bi-GRU
        self.s_net_u = SNet(self.s_count, self.s_length, config.self_attention_hidden_size, config.gru_hidden_size * 2)
        self.s_net_i = SNet(self.s_count, self.s_length, config.self_attention_hidden_size, config.gru_hidden_size * 2)

        self.linear_u = nn.Linear(config.gru_hidden_size * 4, config.gru_hidden_size * 2, bias=False)
        self.linear_i = nn.Linear(config.gru_hidden_size * 4, config.gru_hidden_size * 2, bias=False)

    def forward(self, user_reviews, item_reviews):
        user_reviews = user_reviews.view(-1, self.s_count * self.s_length)
        item_reviews = item_reviews.view(-1, self.s_count * self.s_length)
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


class ControlNet(nn.Module):

    def __init__(self, config, word_emb):
        super(ControlNet, self).__init__()
        self.r_count = config.sent_count
        self.r_length = config.sent_length
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))

    def forward(self, user_reviews, item_reviews, ui_review):
        user_reviews = user_reviews.view(-1, self.r_count * self.r_length)
        item_reviews = item_reviews.view(-1, self.r_count * self.r_length)
        user_emb = self.embedding(user_reviews)
        item_emb = self.embedding(item_reviews)
        ui_emb = self.embedding(ui_review)  # shape(batch_size, review_length, word_dim)
        # todo


class UMPR(nn.Module):

    def __init__(self, config, word_emb):
        super(UMPR, self).__init__()
        self.review_net = ReviewNet(config, word_emb)

        self.linear_fusion = nn.Sequential(
            nn.Linear(config.gru_hidden_size * 2, 1),
            nn.ReLU()
        )

    def forward(self, user_reviews, item_reviews, ui_reviews):
        review_net_represent = self.review_net(user_reviews, item_reviews)

        prediction = self.linear_fusion(review_net_represent)
        return prediction.squeeze()
