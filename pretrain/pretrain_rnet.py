import argparse
import os
import random
import sys
from collections import defaultdict

import gensim
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('../')
from src.helpers import get_logger
from src.model import RNet
from src.word2vec import Word2vec
from pretrain.abae import ABAEDataset, train_ABAE, ABAE


def rand_nums(arr, k=1, except_num=None):
    x = random.sample(arr, k + 1)
    x = [i for i in x if i != except_num][:k]
    if len(x) == 1:
        return x[0]
    return x


class PretrainRNet(nn.Module):
    def __init__(self, word2vec, gru_hidden):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word2vec.embedding))
        self.r_net = RNet(word2vec.word_dim, gru_hidden)
        self.linear = nn.Sequential(
            nn.Linear(gru_hidden * 4, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()

    def forward(self, u, u_length, i, i_length, target):
        device = self.embedding.weight.device
        # u shape(batch_size,sent_length); u_length shape(batch_size)
        u, i, target = [d.to(device) for d in (u, i, target)]
        u = u.view(u.shape[0], 1, u.shape[1])
        i = i.view(i.shape[0], 1, i.shape[1])
        u_length = u_length.view(u_length.shape[0], 1)
        i_length = i_length.view(i_length.shape[0], 1)
        u = self.embedding(u)
        i = self.embedding(i)
        _, _, _, _, att_u, att_i = self.r_net(u, i, u_length, i_length)
        att = torch.cat([att_u, att_i], dim=-1)
        result = self.linear(att).squeeze(-1)
        loss = self.loss_fn(result, target)
        return result, loss

    def save_r_net(self, save_path):
        torch.save(self.r_net, save_path)


class PretrainRNetDataset(torch.utils.data.Dataset):
    def __init__(self, word2vec, sentences, trained_abae, max_length=20):
        data = [word2vec.sent2indices(sent)[:max_length] for sent in sentences]

        # Extract aspect for each sentence
        ABAE_dataloader = ABAEDataset(word2vec, trains, training=False, max_length=max_length)
        ABAE_dataloader = DataLoader(ABAE_dataloader, batch_size=1024)
        sent_aspect = []
        with torch.no_grad():
            trained_abae.eval()
            for batch in ABAE_dataloader:
                probs = trained_abae(batch[0])
                pred = probs.max(dim=-1)[1]
                sent_aspect.extend(pred.cpu().numpy())

        # category by aspect
        category = defaultdict(list)
        for sent, label in zip(data, sent_aspect):
            category[label].append(sent)
        category = dict((k, v) for k, v in category.items() if len(v) >= 5)  # Remove category with too few sentences.

        # samples
        sample1, length1, sample2, length2, labels = [], [], [], [], []
        for cate, sents in category.items():
            for i, sent in enumerate(sents):
                sent_pos = sents[rand_nums(range(len(sents)), except_num=i)]  # positive sample
                sample1.append(word2vec.pad(sent, max_length))
                length1.append(len(sent))
                sample2.append(word2vec.pad(sent_pos, max_length))
                length2.append(len(sent_pos))
                labels.append(1)

                neg_sents = category[rand_nums(range(len(category)), except_num=cate)]
                sent_neg = neg_sents[rand_nums(range(len(neg_sents)), except_num=i)]  # negtive sample
                sample1.append(word2vec.pad(sent, max_length))
                length1.append(len(sent))
                sample2.append(word2vec.pad(sent_neg, max_length))
                length2.append(len(sent_neg))
                labels.append(0)

        self.data = (
            torch.LongTensor(sample1),
            torch.LongTensor(length1),
            torch.LongTensor(sample2),
            torch.LongTensor(length2),
            torch.FloatTensor(labels),
        )

    def __getitem__(self, idx):
        return tuple(x[idx] for x in self.data)

    def __len__(self):
        return len(self.data[0])


def pretrain_r_net(word2vec, train_dataset, trained_abae, save_r_net_path, args):
    logger.info('Start to pretrain R-Net.')

    logger.info('Load dataloader for pretraining R-Net')
    train_data = PretrainRNetDataset(w2v, train_dataset, trained_abae, max_length=args.max_length)
    train_dlr = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)

    pretrain_r = PretrainRNet(word2vec, gru_hidden=args.gru_size).to(args.device)
    opt = torch.optim.Adam(pretrain_r.parameters(), args.learning_rate, weight_decay=args.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, args.lr_decay)

    logger.info('Start to train.')
    for epoch in range(args.train_epochs):
        pretrain_r.train()
        total_loss, total_samples = 0, 0
        for batch in tqdm(train_dlr, desc=f'pretraining for R-Net epoch {epoch}'):
            _, loss = pretrain_r(*batch)
            loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(_)
            total_samples += len(_)

        lr_sch.step()
        train_loss = total_loss / total_samples
        logger.info(f"Epoch {epoch:3d}; train loss {train_loss:.6f}")
    logger.info(f"End of Training. Then save R-Net to disk.")
    pretrain_r.save_r_net(save_r_net_path)


if __name__ == '__main__':
    logger = get_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--l2_regularization', type=float, default=1e-6)
    parser.add_argument('--lr_decay', type=float, default=0.99)
    parser.add_argument('--train_w2v', type=bool, default=False, help='train word2vec by gensim')
    parser.add_argument('--word2vec', type=str, default=os.path.join(sys.path[0], './dataset/word2vec/music_small'))
    parser.add_argument('--vocab_size', type=int, default=9000, help='max size of vocab')
    parser.add_argument('--max_length', type=int, default=20, help='max length of sentence')
    parser.add_argument('--aspect_size', type=int, default=14, help='Aspect size.')
    parser.add_argument('--data_dir', type=str, default=os.path.join(sys.path[0], '../data/music_small'))
    parser.add_argument('--gru_size', type=int, default=64, help='GRU size of R-Net, equal to UMPR does.')
    parser.add_argument('--save_ABAE', type=str, default=os.path.join(sys.path[0], './model/trained_ABAE.pt'))
    parser.add_argument('--save_rnet', type=str, default=os.path.join(sys.path[0], './model/pretraining_rnet.pt'))
    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, 'train.csv')
    # valid_path = os.path.join(args.data_dir, 'valid.csv')
    # test_path = os.path.join(args.data_dir, 'test.csv')

    logger.debug('Load sentences')
    trains = pd.read_csv(train_path)['review'].to_list()
    trains = [sent for review in trains for sent in str(review).split('.') if len(sent) > 10]

    logger.debug('Load word embedding...')
    if args.train_w2v:
        logger.info('Train word2vec using gensim.')
        wv = gensim.models.Word2Vec([s.split() for s in trains], size=200, window=5, min_count=10, workers=4)
        os.makedirs(os.path.dirname(args.word2vec), exist_ok=True)
        wv.save(args.word2vec)
    w2v = Word2vec(args.word2vec, source='gensim', vocab_size=args.vocab_size)

    logger.debug('Load trained ABAE.')
    if not os.path.exists(args.save_ABAE):
        logger.info(f'Start to train ABAE! No such file "{args.save_ABAE}".')
        train_ABAE(w2v, trains, sent_len=20, neg_count=20, batch_size=512, aspect_size=args.aspect_size,
                   abae_regular=0.1, device=args.device,
                   learning_rate=0.001, lr_decay=0.99, train_epochs=10, save_path=args.save_ABAE, logger=logger)

    trained_ABAE = torch.load(args.save_ABAE)
    pretrain_r_net(w2v, trains, trained_ABAE, args.save_rnet, args)
