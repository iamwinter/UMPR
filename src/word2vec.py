import gensim
import numpy
from tqdm import tqdm


class Word2vec:
    def __init__(self, emb_path, source='glove', vocab_size=0):
        assert source in ['glove', 'gensim'], 'Please set embedding source name correctly'
        self.padding = '<PAD>'
        self.unknown = '<UNK>'
        self.number = '<NUM>'
        self.vocab = [self.padding, self.unknown, self.number]
        self.word2index = dict({self.padding: 0, self.unknown: 1, self.number: 2})
        self.embedding = [numpy.array([])] * 3
        if source == 'glove':
            self._from_glove(emb_path)
        if source == 'gensim':
            self._from_gensim(emb_path, vocab_size)
        for i in range(3):
            self.embedding[i] = numpy.zeros_like(self.embedding[3])
        self.word_dim = len(self.embedding[0])

    def sent2indices(self, sentence, align_length=0):
        indices = list()
        for w in sentence.replace('.', ' ').strip().split():
            if w.isdigit():
                indices.append(self.word2index[self.number])
            elif w in self.word2index:
                indices.append(self.word2index[w])
            else:
                indices.append(self.word2index[self.unknown])
            if 0 < align_length <= len(indices):
                break
        if 0 < align_length and len(indices) < align_length:
            indices += [self.word2index[self.padding]] * (align_length - len(indices))
        return indices

    def pad(self, sequence, pad_length):
        if len(sequence) == pad_length:
            return sequence
        elif len(sequence) < pad_length:
            return sequence + [self.word2index[self.padding]] * (pad_length - len(sequence))
        else:
            return sequence[:pad_length]

    def _from_glove(self, emb_path):
        with open(emb_path, encoding='utf-8') as f:
            for line in tqdm(f.readlines(), desc=f'Loading word2vec from {emb_path}', leave=False):
                tokens = line.split()
                self.vocab.append(tokens[0])
                self.word2index[tokens[0]] = len(self.word2index)
                self.embedding.append(numpy.array([float(i) for i in tokens[1:]]))

    def _from_gensim(self, emb_path, vocab_size):
        model = gensim.models.Word2Vec.load(emb_path)
        vocabs = model.wv.vocab.items()
        if vocab_size > 0:
            vocabs = sorted(vocabs, key=lambda x: x[1].count, reverse=True)[:vocab_size]  # sort by frequency
        for w, _ in tqdm(vocabs, desc=f'Loading word2vec from {emb_path}', leave=False):
            self.vocab.append(w)
            self.word2index[w] = len(self.word2index)
            self.embedding.append(model.wv[w])
            if 0 < vocab_size <= len(self.embedding):
                break

    def __len__(self):
        return len(self.embedding)
