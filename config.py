import inspect
import argparse
import torch


class Config:
    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    train_epochs = 10
    batch_size = 128
    learning_rate = 0.0001
    l2_regularization = 1e-6
    learning_rate_decay = 0.99

    review_count = 10
    review_length = 30
    lowest_review_count = 2
    PAD_WORD = '<UNK>'

    gru_hidden_size = 64  # R-net. It's u in paper
    self_attention_hidden_size = 64  # S-net. It's us in paper

    def __init__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))

        parser = argparse.ArgumentParser()
        for key, val in attributes:
            parser.add_argument('--' + key, dest=key, type=type(val), default=val)
        for key, val in parser.parse_args().__dict__.items():
            self.__setattr__(key, val)
