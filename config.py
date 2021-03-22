import inspect
import argparse
import torch


class Config:
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    train_epochs = 20
    batch_size = 32
    learning_rate = 0.00005
    l2_regularization = 1e-3
    learning_rate_decay = 0.99

    word2vec_file = 'embedding/glove.6B.50d.txt'
    data_dir = 'data/music'

    PAD_WORD = '<UNK>'
    sent_count = 30  # number of sentence per user/item
    lowest_sent_count = 10
    ui_sent_count = 5  # number of sentence in a review that u to i
    sent_length = 20  # length of per sentence
    min_photo_count = 1

    gru_size = 64  # R-net. 64. It's u in paper
    self_atte_size = 64  # S-net. 64. It's us in paper
    kernel_count = 120  # For CNN of C-net. 120
    kernel_size = 3  # For CNN of C-net. 原文说该值=1 2 3分别对应40个filters，共120个filters
    view_size = 10  # multi-view size of C-net. 原文没说最佳值
    threshold = 0.35  # threshold of C-net
    loss_v_rate = 0.1  # rate of loss_v

    def __init__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))

        parser = argparse.ArgumentParser()
        for key, val in attributes:
            parser.add_argument('--' + key, dest=key, type=type(val), default=val)
        for key, val in parser.parse_args().__dict__.items():
            self.__setattr__(key, val)

    def __str__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))
        to_str = ''
        for key, val in attributes:
            to_str += '{} = {}\n'.format(key, val)
        return to_str
