import argparse
import os
import random
import sys

import gensim
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.helpers import get_logger
from src.word2vec import Word2vec


class ABAEDataset(torch.utils.data.Dataset):
    def __init__(self, word2vec, sentences, training=True, labels=None, max_length=20, neg_count=20):
        data = [word2vec.sent2indices(sent, align_length=max_length) for sent in sentences]

        if not training:
            if labels is not None:  # test
                self.data = (torch.LongTensor(data), labels)
            else:  # predict
                self.data = [torch.LongTensor(data), ]
        else:
            pos, neg = [], []
            for i, s in enumerate(data):
                pos.append(s)
                neg_idx = [idx for idx in random.sample(range(len(data)), k=neg_count + 1) if i != idx][:neg_count]
                neg.append([data[idx] for idx in neg_idx])
            self.data = (torch.LongTensor(pos), torch.LongTensor(neg))

    def __getitem__(self, idx):
        return tuple(x[idx] for x in self.data)

    def __len__(self):
        return len(self.data[0])


class ABAE(nn.Module):
    def __init__(self, word_emb, aspect_size, reg_rate):
        super().__init__()
        self.eps = 1e-6
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

        pos_emb = self.embedding(pos.to(device))
        ys = pos_emb.sum(dim=-2)  # pos(batch_size,emb_size)
        di = pos_emb @ self.M @ ys.unsqueeze(-1)  # (batch_size,seq_len,1)
        ai = di.squeeze(-1).softmax(dim=-1)  # (batch_size,seq_len)

        pos_zs = ai.unsqueeze(-2) @ pos_emb  # (batch_size,1,emb_size)
        pos_pt = self.fc(pos_zs)  # (batch_size,1,aspect_size)
        pos_rs = pos_pt @ self.aspect  # (batch_size,1,emb_size)
        pos_zs = pos_zs.squeeze(-2)
        pos_pt = pos_pt.squeeze(-2)
        pos_rs = pos_rs.squeeze(-2)

        if neg is None:  # Test mode
            return pos_pt

        neg_emb = self.embedding(neg.to(device))
        neg_zs = neg_emb.sum(dim=-2)

        # Calculate loss
        normed_pos_zs = pos_zs / (self.eps + pos_zs.norm(dim=-1, keepdim=True))
        normed_pos_rs = pos_rs / (self.eps + pos_rs.norm(dim=-1, keepdim=True))
        normed_neg_zs = neg_zs / (self.eps + neg_zs.norm(dim=-1, keepdim=True))
        loss = 1 - (normed_pos_rs * normed_pos_zs).sum(dim=-1, keepdim=True) \
               + (normed_pos_rs.unsqueeze(-2) * normed_neg_zs).sum(dim=-1)
        loss = torch.relu(loss).mean()

        normed_aspect = self.aspect / (self.eps + self.aspect.norm(dim=-1, keepdim=True))
        penalty = normed_aspect @ normed_aspect.transpose(0, 1) - torch.eye(self.aspect.shape[0]).to(device)
        loss += self.reg_rate * penalty.norm()
        return pos_pt, loss

    def get_aspect_words(self, top=10):
        aspects = []
        emb = self.embedding.weight.detach()
        normed_embedding = emb / (self.eps + emb.norm(dim=-1, keepdim=True))
        for i, asp_emb in enumerate(self.aspect.detach()):
            normed_asp_emb = asp_emb / (self.eps + asp_emb.norm(dim=-1, keepdim=True))
            sims = (normed_embedding * normed_asp_emb).sum(dim=-1)
            ordered_words = sims.argsort(dim=-1, descending=True)[:top]
            aspects.append(ordered_words)
        return aspects


def train_ABAE(word2vec, train_data, sent_len, neg_count, batch_size, aspect_size,
               abae_regular, device, learning_rate,
               lr_decay, train_epochs, save_path, valid=None, logger=None):
    logger.info('Loading training dataset')

    train_data = ABAEDataset(word2vec, train_data, max_length=sent_len, neg_count=neg_count)
    train_dlr = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    if valid is not None:
        valid_data = ABAEDataset(w2v, valid, training=True, max_length=sent_len, neg_count=1)  # valid
        valid_dlr = DataLoader(valid_data, batch_size=args.batch_size * 2)
    else:
        valid_dlr = []

    model = ABAE(word2vec.embedding, aspect_size, abae_regular).to(device)
    opt = torch.optim.Adam(model.parameters(), learning_rate)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)

    logger.info('Start to train.')
    for epoch in range(train_epochs):
        model.train()
        total_loss, total_samples = 0, 0
        for batch in tqdm(train_dlr, desc=f'ABAE training epoch {epoch}'):
            label_probs, loss = model(*batch)
            loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(label_probs)
            total_samples += len(label_probs)

        lr_sch.step()
        train_loss = total_loss / total_samples
        out_info = f"Epoch {epoch:3d}; train loss {train_loss:.6f}; "
        # Valid
        if valid is not None:
            with torch.no_grad():
                model.eval()
                total_loss, total_samples = 0, 0
                category = [0] * aspect_size
                for i, batch in enumerate(valid_dlr):
                    probs, valid_loss = model(*batch)
                    probs = probs.max(dim=-1)[1]
                    for c in probs:
                        category[c.item()] += 1
                    valid_loss = valid_loss.mean()
                    total_loss += valid_loss.item() * len(probs)
                    total_samples += len(probs)
                total_loss /= total_samples
                out_info += f'Valid loss: {total_loss:.6f}; number of per category:{category}'
        logger.info(out_info)

    for i, ap in enumerate(model.get_aspect_words(10)):
        logger.debug(f'Aspect: {i}: {[word2vec.vocab[k] for k in ap]}')

    if hasattr(model, 'module'):
        torch.save(model.module, save_path)
    else:
        torch.save(model, save_path)
    logger.info(f'Trained model "{save_path}" has been saved.')


def evaluate(model, word2vec, tests, test_labels, sent_len=20, batch_size=1024):
    test_data = ABAEDataset(word2vec, tests, training=False, labels=test_labels, max_length=sent_len)
    test_dlr = DataLoader(test_data, batch_size=batch_size)

    # aspect_words was made up according to trained aspect embedding.
    aspect_words = []
    categories = ['Food', 'Staff', 'Ambience', 'Price', 'Anecdotes', 'Miscellaneous']
    logger.info(f'Please choose a category from following list for each aspect.')
    logger.info(dict(enumerate(categories)))
    for k in range(len(model.aspect)):
        aspect_words.append(categories[int(input(f'Input index(0~{len(categories) - 1}) to aspect {k}:'))])

    correct, sample_count = 0, 0
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_dlr, desc='Evaluate'):
            probs = model(batch[0])
            pred = probs.max(dim=-1)[1]
            for truth, aid in zip(batch[-1], pred.cpu().numpy()):
                if truth == aspect_words[aid]:
                    correct += 1
            sample_count += len(probs)
    logger.info(f'Accuracy: {correct / sample_count:.6f}')


if __name__ == '__main__':
    logger = get_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train_epochs', type=int, default=15, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--abae_regular', type=float, default=0.1)
    parser.add_argument('--lr_decay', type=float, default=0.99)
    parser.add_argument('--data_dir', type=str, default='dataset/restaurant', help='dataset location')
    parser.add_argument('--train_w2v', type=bool, default=False, help='train word2vec by gensim')
    parser.add_argument('--vocab_size', type=int, default=9000, help='max size of vocab')
    parser.add_argument('--max_length', type=int, default=20, help='max length of sentence')
    parser.add_argument('--neg_count', type=int, default=20, help='how many negative sample for a positive one.')
    parser.add_argument('--aspect_size', type=int, default=14, help='Aspect size.')
    parser.add_argument('--save_path', type=str, default=os.path.join(sys.path[0], 'model/ABAE.pt'))
    args = parser.parse_args()

    word2vec_path = os.path.join(sys.path[0], args.data_dir, 'w2v_embedding')
    train_path = os.path.join(sys.path[0], args.data_dir, 'train.txt')
    test_path = os.path.join(sys.path[0], args.data_dir, 'test.txt')
    test_label_path = os.path.join(sys.path[0], args.data_dir, 'test_label.txt')
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    trains = open(train_path, 'r').readlines()
    tests = open(test_path, 'r').readlines()
    test_label = [s.strip() for s in open(test_label_path, 'r').readlines()]
    logger.info(f'train sentences: {len(trains)}')
    logger.info(f'test sentences: {len(tests)}')

    if args.train_w2v:
        wv = gensim.models.Word2Vec([s.split() for s in trains + tests], size=200, window=5, min_count=10, workers=4)
        wv.save(word2vec_path)

    w2v = Word2vec(word2vec_path, source='gensim', vocab_size=args.vocab_size)
    logger.info(f'vocabulary size: {len(w2v)}')

    train_ABAE(w2v, trains, args.max_length, args.neg_count, args.batch_size, args.aspect_size, args.abae_regular,
               args.device, args.learning_rate, args.lr_decay, args.train_epochs, args.save_path, tests, logger)

    # Evaluate
    test_model = torch.load(args.save_path)
    evaluate(test_model, w2v, tests, test_label, sent_len=args.max_length, batch_size=1024)
