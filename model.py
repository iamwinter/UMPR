import torch
from torch import nn


class RNet(nn.Module):

    def __init__(self, gru_in, gru_out):
        super(RNet, self).__init__()
        self.gru = nn.GRU(input_size=gru_in, hidden_size=gru_out, batch_first=True, bidirectional=True)
        self.M = nn.Parameter(torch.randn(2 * gru_out, 2 * gru_out))

    def forward(self, user_emb, item_emb):
        gru_u, hn = self.gru(user_emb)
        gru_i, hn = self.gru(item_emb)  # out(batch_size, sent_count * sent_length, 2*gru_out)
        A = gru_i @ self.M @ gru_u.transpose(-1, -2)
        soft_u = torch.softmax(torch.max(A, dim=-2).values, dim=-1)  # column
        soft_i = torch.softmax(torch.max(A, dim=-1).values, dim=-1)  # row. out(batch, sent_count * sent_length)
        atte_u = gru_u.transpose(-1, -2) @ soft_u.unsqueeze(-1)
        atte_i = gru_i.transpose(-1, -2) @ soft_i.unsqueeze(-1)  # shape(batch_size, 2*gru_out, 1)
        return gru_u.contiguous(), gru_i.contiguous(), soft_u, soft_i, atte_u.squeeze(-1), atte_i.squeeze(-1)


class SNet(nn.Module):

    def __init__(self, sent_count, sent_length, self_atte_size, repr_size):
        super(SNet, self).__init__()
        self.s_count = sent_count
        self.s_length = sent_length

        self.Ms = nn.Parameter(torch.randn(self_atte_size, repr_size))  # repr_size = 2u in the paper
        self.Ws = nn.Parameter(torch.randn(1, self_atte_size))

    def forward(self, reviews, gru_hidden, word_soft):
        # self-attention for sentence-level sentiment.
        batch_size = reviews.shape[0]
        temp_batch_size = batch_size * self.s_count
        gru_hidden = gru_hidden.reshape(temp_batch_size, self.s_length, -1).transpose(-1, -2)
        sent_soft = torch.softmax(self.Ws @ torch.tanh(self.Ms @ gru_hidden), dim=-1)  # (temp_batch,1,r_length)
        self_atte = gru_hidden @ sent_soft.transpose(-1, -2)  # out(temp_batch, repr_size, 1)

        sentiment_emb = word_soft.view(temp_batch_size, -1).sum(dim=-1, keepdim=True) * self_atte.squeeze(-1)
        sentiment_emb = sentiment_emb.view(batch_size, self.s_count, -1).sum(dim=-2)
        return self_atte.view(batch_size, self.s_count, -1), sentiment_emb  # output(batch, repr_size)


class CNet(nn.Module):

    def __init__(self, sent_count, sent_length, gru_in, gru_out, k_count, k_size, view_size, threshold=0.35):
        super(CNet, self).__init__()
        self.s_count = sent_count
        self.s_length = sent_length
        self.threshold = threshold

        self.gru = nn.GRU(input_size=gru_in, hidden_size=gru_out, batch_first=True, bidirectional=True)  # BiGRU
        self.cnn = nn.Sequential(
            # permute(0,2,1) -> (temp_bs, 2*gru_out, s_length)
            nn.Conv1d(in_channels=2 * gru_out, out_channels=k_count, kernel_size=k_size, padding=(k_size - 1) // 2),
            # (temp_bs, k_count, s_length)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=sent_length),
            # (temp_bs, k_count, 1)
        )
        # shape -> (batch_size, sent_count, k_count)
        self.linear = nn.Sequential(
            nn.Linear(k_count, view_size),
            nn.Sigmoid()
            # out(batch_size, sent_count, view_size)
        )

    def forward(self, review_emb):
        gru_repr, hn = self.gru(review_emb)  # shape(batch, s_count*s_length, 2*gru_out)
        temp_bs = review_emb.shape[0] * self.s_count
        gru_repr = gru_repr.reshape(temp_bs, self.s_length, -1).transpose(-1, -2)
        cnn_out = self.cnn(gru_repr)
        cnn_out = cnn_out.view(-1, self.s_count, cnn_out.shape[1])

        linear_out = self.linear(cnn_out)
        linear_out[linear_out < self.threshold] = 0
        final_repr = torch.sum(linear_out ** 2, dim=-2)  # out(batch_size, view_size)
        return gru_repr, linear_out, final_repr


class ReviewNet(nn.Module):

    def __init__(self, s_count, s_length, emb_size, gru_size, atte_size):
        super(ReviewNet, self).__init__()
        self.r_net = RNet(emb_size, gru_size)  # Note: using Bi-GRU
        self.s_net_u = SNet(s_count, s_length, atte_size, gru_size * 2)
        self.s_net_i = SNet(s_count, s_length, atte_size, gru_size * 2)

        self.linear_u = nn.Linear(gru_size * 4, gru_size * 2, bias=False)
        self.linear_i = nn.Linear(gru_size * 4, gru_size * 2, bias=False)

    def forward(self, user_emb, item_emb):
        gru_u, gru_i, soft_u, soft_i, atte_u, atte_i = self.r_net(user_emb, item_emb)
        _, sentiment_u = self.s_net_u(user_emb, gru_u, soft_u)
        _, sentiment_i = self.s_net_i(item_emb, gru_i, soft_i)

        # Textual Matching
        repr_u = torch.cat([atte_u, sentiment_u], dim=-1)  # formula (7)
        repr_i = torch.cat([atte_i, sentiment_i], dim=-1)
        represent = torch.tanh(self.linear_u(repr_u) + self.linear_i(repr_i))  # formula (8)
        return represent  # output shape(batch, 2u) where u is GRU hidden size


class ControlNet(nn.Module):
    def __init__(self, sent_count, sent_length, emb_size, gru_size, k_count, k_size, view_size, threshold, atte_size):
        super(ControlNet, self).__init__()
        self.s_count = sent_count
        self.s_length = sent_length

        self.c_net = CNet(sent_count, sent_length, emb_size, gru_size, k_count, k_size, view_size, threshold)
        self.s_net = SNet(sent_count, sent_length, atte_size, repr_size=gru_size * 2)
        self.ss_net = nn.Sequential(
            nn.Linear(gru_size * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, review_emb):
        gru_repr, view_possibility, c_net_out = self.c_net(review_emb)
        s, _ = self.s_net(review_emb, gru_repr, view_possibility)
        senti_score = self.ss_net(s)  # out(batch_size, s_count, 1)
        view_score = (senti_score * (view_possibility ** 2)).sum(dim=-2) / (view_possibility ** 2).sum(dim=-2)  # (17)
        q_p = torch.zeros_like(view_score)
        q_pos = 4 * (view_score - 0.5) ** 2
        q_neg = 4 * (0.5 - view_score) ** 2
        q_p[view_score > 0.5] = 1
        q_pos[view_score < 0.5] = 0
        q_neg[view_score > 0.5] = 0

        prefer_pos = c_net_out * q_p * q_pos
        prefer_neg = c_net_out * (1 - q_p) * q_neg
        return prefer_pos, prefer_neg  # (batch_size, view_size)


class UMPR(nn.Module):
    def __init__(self, config, word_emb):
        super(UMPR, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))

        self.review_net = ReviewNet(config.sent_count, config.sent_length, self.embedding.embedding_dim,
                                    config.gru_size, config.self_atte_size)
        self.control_net = ControlNet(config.ui_sent_count, config.sent_length, self.embedding.embedding_dim,
                                      config.gru_size, config.kernel_count, config.kernel_size, config.view_size,
                                      config.threshold, config.self_atte_size)

        self.linear_fusion = nn.Sequential(
            nn.Linear(config.gru_size * 2, 1),
            nn.ReLU()
        )

    def forward(self, user_reviews, item_reviews, ui_reviews):
        sent_count = user_reviews.shape[-2]
        ui_sent_count = ui_reviews.shape[-2]
        sent_length = user_reviews.shape[-1]
        user_reviews = user_reviews.view(-1, sent_count * sent_length)
        item_reviews = item_reviews.view(-1, sent_count * sent_length)
        ui_reviews = ui_reviews.view(-1, ui_sent_count * sent_length)
        user_emb = self.embedding(user_reviews)  # (batch_size, sent_count * sent_length, emb_size)
        item_emb = self.embedding(item_reviews)
        ui_emb = self.embedding(ui_reviews)

        review_net_repr = self.review_net(user_emb, item_emb)
        prefer_pos, prefer_neg = self.control_net(ui_emb)

        prediction = self.linear_fusion(review_net_repr)
        return prediction.squeeze(-1)
