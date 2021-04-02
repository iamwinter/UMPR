import argparse
import os
import random
import sys

import gensim
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader

from src.helpers import get_logger, process_bar
from src.word2vec import Word2vec


class ABAE(nn.Module):
    def __init__(self, word_emb, aspect_size, reg_rate):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.embedding.weight.requires_grad_()
        self.M = nn.Parameter(torch.randn([self.embedding.embedding_dim] * 2))
        self.fc = nn.Sequential(
            nn.Linear(self.embedding.embedding_dim, aspect_size),
            nn.Softmax(dim=-1)
        )
        km = KMeans(n_clusters=aspect_size)
        km.fit(word_emb)
        centers = km.cluster_centers_
        self.aspect = nn.Parameter(torch.Tensor(centers))
        self.reg_rate = reg_rate

    def forward(self, pos, neg=None):
        device = self.embedding.weight.device
        pos = pos.to(device)
        pos_zs, pos_pt, pos_rs = self._forward(pos)

        if neg is None:  # Test mode
            return pos_pt

        neg = neg.to(device)
        neg_zs = self._forward(neg, negative=True)
        loss = torch.max(torch.zeros(1).to(device), 1 - pos_rs * pos_zs + pos_rs * neg_zs)
        normed_aspect = self.aspect / self.aspect.norm(dim=-1, keepdim=True)
        penalty = torch.abs(normed_aspect @ normed_aspect.transpose(0, 1) - torch.eye(self.aspect.shape[0]).to(device))
        loss += self.reg_rate * penalty.norm()
        return pos_pt, loss

    def _forward(self, sentences, negative=False):
        emb = self.embedding(sentences)
        ys = emb.sum(dim=-2).unsqueeze(-1)  # (batch_size,emb_size,1)
        if negative:
            return ys.squeeze(-1)
        di = emb @ self.M @ ys  # (batch_size,seq_len,1)
        ai = di.squeeze(-1).softmax(dim=-1)  # (batch_size,seq_len)

        zs = ai.unsqueeze(-2) @ emb  # (batch_size,1,emb_size)
        pt = self.fc(zs)  # (batch_size,1,aspect_size)
        rs = pt @ self.aspect  # (batch_size,1,aspect_size)
        return zs.squeeze(-2), pt.squeeze(-2), rs.squeeze(-2)

    def get_aspect_words(self, top=10):
        aspects = []
        for i, asp_emb in enumerate(self.aspect.detach()):
            sims = (self.embedding.weight.detach() * asp_emb).sum(dim=-1)
            ordered_words = sims.argsort(dim=-1, descending=True)[:top]
            aspects.append(ordered_words)
        return aspects


class ABAEDataset(torch.utils.data.Dataset):
    def __init__(self, word2vec, sentences, labels=None, max_length=20, neg_count=20):
        data = [word2vec.sent2indices(sent, align_length=max_length) for sent in sentences]

        if labels is not None:  # test mode
            self.data = (torch.LongTensor(data), labels)
        else:
            pos, neg = [], []
            for i, s in enumerate(data):
                neg_idx = [idx for idx in random.sample(range(len(data)), k=neg_count + 1) if i != idx][:neg_count]
                for idx in neg_idx:
                    pos.append(s)
                    neg.append(data[idx])
            self.data = (torch.LongTensor(pos), torch.LongTensor(neg))

    def __getitem__(self, idx):
        return tuple(x[idx] for x in self.data)

    def __len__(self):
        return len(self.data[0])


def train_ABAE(word2vec, train_data, args, model_path, logger=get_logger()):
    logger.info('Loading training dataset')

    train_data = ABAEDataset(word2vec, train_data, max_length=args.max_length, neg_count=args.neg_count)
    train_dlr = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    model = ABAE(word2vec.embedding, args.aspect_size, args.l2_regularization).to(args.device)
    opt = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, args.learning_rate_decay)

    logger.info('Start to train.')
    for epoch in range(args.train_epochs):
        total_loss, total_samples = 0, 0
        for i, batch in enumerate(train_dlr):
            process_bar(i + 1, len(train_dlr), prefix=f'ABAE training epoch {epoch}')
            model.train()
            label_probs, loss = model(*batch)
            loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(label_probs)
            total_samples += len(label_probs)

        lr_sch.step()
        train_loss = total_loss / total_samples
        logger.info(f"Epoch {epoch:3d}; train loss {train_loss:.6f}")

        if __name__ == '__main__':
            for i, ap in enumerate(model.get_aspect_words(10)):
                logger.debug(f'Aspect: {i}: {[word2vec.vocab[k] for k in ap]}')

        if hasattr(model, 'module'):
            torch.save(model.module, model_path)
        else:
            torch.save(model, model_path)


def test_ABAE(model, test_dlr):
    # aspect_words was made up according to trained aspect embedding.
    aspect_words = {0: 'Food', 1: 'Miscellaneous', 2: 'Miscellaneous', 3: 'Food', 4: 'Miscellaneous',
                    5: 'Food', 6: 'Price', 7: 'Miscellaneous', 8: 'Staff',
                    9: 'Food', 10: 'Food', 11: 'Anecdotes',
                    12: 'Ambience', 13: 'Staff'}
    correct, sample_count = 0, 0
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(test_dlr):
            process_bar(i + 1, len(test_dlr), prefix='Evaluate')
            probs = model(batch[0])
            pred = probs.max(dim=-1)[1]
            for truth, aid in zip(batch[-1], pred.cpu().numpy()):
                if truth == aspect_words[aid]:
                    correct += 1
            sample_count += len(probs)
    return correct / sample_count


def load_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data.append(line)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train_epochs', type=int, default=15, help='number of epochs to train (default: 15)')
    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size for training (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--l2_regularization', type=float, default=0.1)
    parser.add_argument('--learning_rate_decay', type=float, default=0.99)
    parser.add_argument('--train_w2v', type=bool, default=False, help='train word2vec by gensim')
    parser.add_argument('--vocab_size', type=int, default=9000, help='max size of vocab')
    parser.add_argument('--max_length', type=int, default=15, help='max length of sentence')
    parser.add_argument('--neg_count', type=int, default=20, help='how many negative sample for a positive one.')
    parser.add_argument('--aspect_size', type=int, default=14, help='Aspect size.')
    args = parser.parse_args()

    word2vec_path = os.path.join(sys.path[0], './dataset/restaurant/w2v_embedding')
    train_path = os.path.join(sys.path[0], 'dataset/restaurant/train.txt')
    test_path = os.path.join(sys.path[0], 'dataset/restaurant/test.txt')
    test_label_path = os.path.join(sys.path[0], 'dataset/restaurant/test_label.txt')
    save_model = os.path.join(sys.path[0], 'model/test_ABAE.pt')
    os.makedirs(os.path.dirname(save_model), exist_ok=True)

    trains, tests, test_label = load_data(train_path), load_data(test_path), load_data(test_label_path)
    print(f'train sentences: {len(trains)}')
    print(f'test sentences: {len(tests)}')

    if args.train_w2v:
        wv = gensim.models.Word2Vec([s.split() for s in trains + tests], size=200, window=5, min_count=10, workers=4)
        wv.save(word2vec_path)

    w2v = Word2vec(word2vec_path, source='gensim', vocab_size=args.vocab_size)
    print(f'vocabulary size: {len(w2v)}')

    test_data = ABAEDataset(w2v, tests, test_label, max_length=args.max_length)
    test_dlr = DataLoader(test_data, batch_size=args.batch_size * 2)

    train_ABAE(w2v, trains, args, save_model)

    acc = test_ABAE(torch.load(save_model), test_dlr)
    print(f'accuracy: {acc:.6f}')
