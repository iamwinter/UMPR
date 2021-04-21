import torch
import torchvision
from torch import nn


class ImprovedRnn(nn.Module):
    def __init__(self, module, *args, **kwargs):
        assert module in (nn.RNN, nn.LSTM, nn.GRU)
        super().__init__()
        self.module = module(*args, **kwargs)

    def forward(self, data, lengths):  # data shape(batch_size, seq_len, input_size)
        if not hasattr(self, '_flattened'):
            self.module.flatten_parameters()
            setattr(self, '_flattened', True)
        bf = self.module.batch_first
        max_len = data.shape[1]
        package = nn.utils.rnn.pack_padded_sequence(data, lengths.cpu(), batch_first=bf, enforce_sorted=False)
        result, hidden = self.module(package)
        result, lens = nn.utils.rnn.pad_packed_sequence(result, batch_first=bf, total_length=max_len)
        return result[package.unsorted_indices], hidden


class RNet(nn.Module):

    def __init__(self, gru_in, gru_out, pretrained: str = None):
        super().__init__()
        self.gru = ImprovedRnn(nn.GRU, input_size=gru_in, hidden_size=gru_out, batch_first=True, bidirectional=True)
        self.M = nn.Parameter(torch.randn(2 * gru_out, 2 * gru_out))
        if pretrained is not None:
            try:
                self.load_state_dict(torch.load(pretrained).state_dict())
            except:
                print(f'Failed to load R-Net pre-trained weights from "{pretrained}"')

    def forward(self, user_emb, item_emb, u_lengths, i_lengths):
        batch_size = user_emb.shape[0]
        sent_count = user_emb.shape[1]
        sent_length = user_emb.shape[2]
        user_emb = user_emb.view(user_emb.shape[0] * user_emb.shape[1], user_emb.shape[2], user_emb.shape[3])
        item_emb = item_emb.view(item_emb.shape[0] * item_emb.shape[1], item_emb.shape[2], item_emb.shape[3])
        u_lengths = u_lengths.view(u_lengths.shape[0] * u_lengths.shape[1])
        i_lengths = i_lengths.view(i_lengths.shape[0] * i_lengths.shape[1])
        gru_u, hn = self.gru(user_emb, u_lengths)
        gru_i, hn = self.gru(item_emb, i_lengths)  # out(batch_size*sent_count, sent_length, 2*gru_out)
        gru_u = gru_u.reshape(batch_size, sent_count * sent_length, -1)
        gru_i = gru_i.reshape(batch_size, sent_count * sent_length, -1)

        A = gru_i @ self.M @ gru_u.transpose(-1, -2)  # (3) affinity matrix
        soft_u = torch.softmax(torch.max(A, dim=-2).values, dim=-1)  # column
        soft_i = torch.softmax(torch.max(A, dim=-1).values, dim=-1)  # row. out(batch, sent_count * sent_length)
        atte_u = gru_u.transpose(-1, -2) @ soft_u.unsqueeze(-1)
        atte_i = gru_i.transpose(-1, -2) @ soft_i.unsqueeze(-1)  # shape(batch_size, 2*gru_out, 1)
        return gru_u.contiguous(), gru_i.contiguous(), soft_u, soft_i, atte_u.squeeze(-1), atte_i.squeeze(-1)


class SNet(nn.Module):

    def __init__(self, self_atte_size, repr_size, pretrained: str = None):
        super().__init__()
        self.Ms = nn.Parameter(torch.randn(self_atte_size, repr_size))  # repr_size = 2u in the paper
        self.Ws = nn.Parameter(torch.randn(1, self_atte_size))
        if pretrained is not None:
            try:
                self.load_state_dict(torch.load(pretrained).state_dict())
            except:
                print(f'Failed to load S-Net pre-trained weights from "{pretrained}"')

    def forward(self, gru_repr, word_soft, sent_length):
        # self-attention for sentence-level sentiment.
        batch_size = gru_repr.shape[0]
        sent_count = gru_repr.shape[1] // sent_length
        gru_repr = gru_repr.reshape(batch_size * sent_count, sent_length, -1).transpose(-1, -2)
        sent_soft = torch.softmax(self.Ws @ torch.tanh(self.Ms @ gru_repr), dim=-1)  # (temp_batch,1,r_length)
        self_atte = gru_repr @ sent_soft.transpose(-1, -2)  # out(temp_batch, repr_size, 1)

        sentiment_emb = word_soft.view(batch_size * sent_count, -1).sum(dim=-1, keepdim=True) * self_atte.squeeze(-1)
        sentiment_emb = sentiment_emb.view(batch_size, sent_count, -1).sum(dim=-2)
        return self_atte.view(batch_size, sent_count, -1), sentiment_emb  # output(batch, repr_size)


class CNet(nn.Module):

    def __init__(self, gru_in, gru_out, k_count, k_size, view_size, threshold=0.35, pretrained: str = None):
        super().__init__()
        self.threshold = threshold

        self.gru = ImprovedRnn(nn.GRU, input_size=gru_in, hidden_size=gru_out, batch_first=True, bidirectional=True)
        self.cnn = nn.Sequential(
            # permute(0,2,1) -> (temp_bs, 2*gru_out, s_length)
            nn.Conv1d(in_channels=2 * gru_out, out_channels=k_count, kernel_size=k_size, padding=(k_size - 1) // 2),
            nn.ReLU(),
            # (temp_bs, k_count, s_length)
        )
        # max -> shape(temp_bs, k_count)
        # shape -> (batch_size, sent_count, k_count)
        self.linear = nn.Sequential(
            nn.Linear(k_count, view_size),
            nn.Sigmoid()
            # out(batch_size, sent_count, view_size)
        )
        if pretrained is not None:
            try:
                self.load_state_dict(torch.load(pretrained).state_dict())
            except:
                print(f'Failed to load S-Net pre-trained weights from "{pretrained}"')

    def forward(self, review_emb, lengths):
        batch_size = review_emb.shape[0]
        sent_count = review_emb.shape[1]
        sent_length = review_emb.shape[2]
        lengths = lengths.view(lengths.shape[0] * lengths.shape[1])
        gru_repr, hn = self.gru(review_emb.view(batch_size * sent_count, sent_length, -1), lengths)
        gru_repr = gru_repr.reshape(batch_size, sent_count * sent_length, -1)

        cnn_in = gru_repr.reshape(batch_size * sent_count, sent_length, -1).transpose(-1, -2)
        cnn_out = self.cnn(cnn_in)
        cnn_out = cnn_out.max(dim=-1)[0]  # (batch_size*sent_count, k_count)
        cnn_out = cnn_out.view(batch_size, sent_count, -1)

        view_p = self.linear(cnn_out)  # (14) view possibility (batch_size, sent_count, view_size)
        view_p = torch.where(view_p < self.threshold, torch.zeros_like(view_p), view_p)  # (15)
        final_repr = torch.sum(view_p ** 2, dim=-2)  # (16) out(batch_size, view_size)
        return gru_repr, view_p, final_repr


class SSNet(nn.Module):
    def __init__(self, input_size, pretrained: str = None):
        super(SSNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Sigmoid()
        )
        if pretrained is not None:
            try:
                self.load_state_dict(torch.load(pretrained).state_dict())
            except:
                print(f'Failed to load SS-Net pre-trained weights from "{pretrained}"')

    def forward(self, sentiment_emb):  # in(batch_size, s_count, input_size)
        return self.linear(sentiment_emb)


class ReviewNet(nn.Module):

    def __init__(self, emb_size, gru_size, atte_size):
        super().__init__()
        self.r_net = RNet(emb_size, gru_size)  # Note: using Bi-GRU
        self.s_net_u = SNet(atte_size, gru_size * 2)
        self.s_net_i = SNet(atte_size, gru_size * 2)

        self.linear_u = nn.Linear(gru_size * 4, gru_size * 2, bias=False)
        self.linear_i = nn.Linear(gru_size * 4, gru_size * 2, bias=False)

    def forward(self, user_emb, item_emb, u_lengths, i_lengths):
        u_s_length = user_emb.shape[-2]
        i_s_length = item_emb.shape[-2]

        gru_u, gru_i, soft_u, soft_i, atte_u, atte_i = self.r_net(user_emb, item_emb, u_lengths, i_lengths)
        _, sentiment_u = self.s_net_u(gru_u, soft_u, u_s_length)
        _, sentiment_i = self.s_net_i(gru_i, soft_i, i_s_length)

        # Textual Matching
        repr_u = torch.cat([atte_u, sentiment_u], dim=-1)  # formula (7)
        repr_i = torch.cat([atte_i, sentiment_i], dim=-1)
        represent = torch.tanh(self.linear_u(repr_u) + self.linear_i(repr_i))  # formula (8)
        return represent  # output shape(batch, 2u) where u is GRU hidden size


class ControlNet(nn.Module):
    def __init__(self, emb_size, gru_size, k_count, k_size, view_size, threshold, atte_size):
        super().__init__()
        self.c_net = CNet(emb_size, gru_size, k_count, k_size, view_size, threshold)
        self.s_net = SNet(atte_size, repr_size=gru_size * 2)
        self.ss_net = SSNet(input_size=gru_size * 2)

    def forward(self, user_emb, item_emb, ui_emb, u_lengths, i_lengths, ui_lengths):
        ui_s_length = ui_emb.shape[-2]

        gru_repr, view_p, c_net_out = self.c_net(ui_emb, ui_lengths)
        _, _, c_u = self.c_net(user_emb, u_lengths)
        _, _, c_i = self.c_net(item_emb, i_lengths)
        s, _ = self.s_net(gru_repr, view_p, ui_s_length)
        senti_score = self.ss_net(s)  # (17) sentiment score of each vocab. out(batch_size, s_count, 1)
        senti_score = senti_score.expand(-1, -1, view_p.shape[-1])  # copy P of each word to every view
        view_score = torch.sum(senti_score * view_p ** 2, dim=-2).div(torch.sum(view_p ** 2, dim=-2) + 1e-4)  # (18)
        q_p = torch.zeros_like(view_score)
        q_pos = 4 * (view_score - 0.5) ** 2
        q_neg = 4 * (0.5 - view_score) ** 2
        q_p[view_score > 0.5] = 1
        q_pos[view_score < 0.5] = 0
        q_neg[view_score > 0.5] = 0

        prefer_pos = c_net_out * q_p * q_pos
        prefer_neg = c_net_out * (1 - q_p) * q_neg
        return c_u, c_i, prefer_pos, prefer_neg  # (batch_size, view_size)


class VisualNet(nn.Module):
    def __init__(self, view_size, vgg_out=1000):
        super().__init__()
        self.vgg16 = nn.Sequential(
            torchvision.models.vgg16(pretrained=True, num_classes=vgg_out),
            # nn.Sigmoid()
        )
        self.pos_v_emb = nn.Parameter(torch.randn(view_size, vgg_out))
        self.neg_v_emb = nn.Parameter(torch.randn(view_size, vgg_out))
        self.linear = nn.Linear(vgg_out, 1)

    def forward(self, images, c_u, c_i):
        batch_size = images.shape[0]
        view_size = images.shape[1]
        photo_count = images.shape[2]
        images = images.view(batch_size * view_size * photo_count, images.shape[3], images.shape[4], images.shape[5])
        img_repr = self.vgg16(images)
        img_repr = img_repr.view(batch_size, view_size, photo_count, -1)
        img_repr = img_repr.mean(dim=-2)  # eq.(10) (b, view_size, vgg_out)

        img_emb = self.linear(img_repr).squeeze(-1)
        pos_emb = self.linear(self.pos_v_emb).squeeze(-1)
        neg_emb = self.linear(self.neg_v_emb).squeeze(-1)
        pos_match = torch.tanh(torch.abs(pos_emb - img_emb))  # eq.(11) \tilde{x^V+}
        neg_match = torch.tanh(torch.abs(neg_emb - img_emb))  # (b,view_size)

        final_pos = c_u * c_i * (1 - pos_match)
        final_neg = c_u * c_i * (1 - neg_match)
        return pos_match, neg_match, final_pos, final_neg


class UMPR(nn.Module):
    def __init__(self, config, word_emb):
        super().__init__()
        self.review_net_only = config.review_net_only
        self.loss_v_rate = config.loss_v_rate
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))

        self.review_net = ReviewNet(self.embedding.embedding_dim, config.gru_size, config.self_atte_size)

        if config.review_net_only:
            self.linear_fusion = nn.Sequential(
                nn.Linear(config.gru_size * 2, 1),
                nn.ReLU()
            )
        else:
            view_size = len(config.views)
            self.control_net = ControlNet(self.embedding.embedding_dim, config.gru_size, config.kernel_count,
                                          config.kernel_size, view_size, config.threshold, config.self_atte_size)
            self.visual_net = VisualNet(view_size)

            self.linear_fusion = nn.Sequential(
                nn.Linear(config.gru_size * 2 + view_size + view_size, 1),
                nn.ReLU()
            )

    def forward(self, user_reviews, item_reviews, ui_reviews, u_lengths, i_lengths, ui_lengths, photos, labels):
        device = self.embedding.weight.device
        user_reviews, item_reviews, ui_reviews = [v.to(device) for v in (user_reviews, item_reviews, ui_reviews)]
        photos, labels = [v.to(device) for v in (photos, labels)]

        user_emb = self.embedding(user_reviews)  # (batch_size, sent_count, sent_length, emb_size)
        item_emb = self.embedding(item_reviews)
        ui_emb = self.embedding(ui_reviews)

        review_net_repr = self.review_net(user_emb, item_emb, u_lengths, i_lengths)  # (batch, 2u) where u is GRU hidden
        if self.review_net_only:
            prediction = self.linear_fusion(review_net_repr).squeeze(-1)
            loss = torch.nn.functional.mse_loss(prediction, labels, reduction='mean')
        else:
            c_u, c_i, prefer_pos, prefer_neg = self.control_net(user_emb, item_emb, ui_emb, u_lengths, i_lengths, ui_lengths)
            pos_match, neg_match, final_pos, final_neg = self.visual_net(photos, c_u, c_i)  # (b,view_size)

            prediction = self.linear_fusion(torch.cat([review_net_repr, final_pos, final_neg], dim=-1)).squeeze(-1)
            loss_r = torch.nn.functional.mse_loss(prediction, labels, reduction='mean')
            loss_v = torch.mean(prefer_pos.transpose(-1, -2) @ pos_match + prefer_neg.transpose(-1, -2) @ neg_match)
            loss = loss_r + loss_v * self.loss_v_rate
        return prediction, loss
