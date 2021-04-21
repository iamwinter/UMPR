import ast
import inspect
import argparse
import torch


class Config:
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    multi_gpu = True  # Whether to use multiple GPUs for training.
    train_epochs = 20
    batch_size = 64
    learning_rate = 1e-6
    l2_regularization = 1e-3
    lr_decay = 0.99

    word2vec_file = 'embedding/glove.6B.50d.txt'
    data_dir = 'data/music'
    log_path = ''
    model_path = ''

    test_only = False  # If it's true, you must set log_path and model_path.
    review_net_only = False  # If it's true, only review net will be executable.

    review_level = 'sentence'  # How to split reviews. Assert it is in ['sentence', 'review']
    max_sent_count = 20  # max number of sentences per user/item
    min_sent_count = 5
    max_ui_sent_count = 5  # max number of sentences u wrote to i
    max_sent_length = 20  # max length per sentence
    views = ['unknown']  # For amazon. Its length is "view_size" of C-net and Visual-Net. 1 for amazon while 4 for yelp!
    # views = ['food', 'inside', 'outside', 'drink']  # yelp
    photo_count = 1  # number of photos for each view

    gru_size = 64  # R-net. 64. It's u in paper
    self_atte_size = 64  # S-net. 64. It's us in paper
    kernel_count = 120  # For CNN of C-net. 120
    kernel_size = 3  # For CNN of C-net. 原文说该值=1 2 3分别对应40个filters，共120个filters
    threshold = 0.35  # threshold of C-net
    loss_v_rate = 0.1  # rate of loss_v

    def __init__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))

        parser = argparse.ArgumentParser()
        for key, val in attributes:
            receive_type = type(val)
            if receive_type in [bool, int, float, list]:
                receive_type = ast.literal_eval  # type can accept a function converting the input to python variable
            parser.add_argument('--' + key, dest=key, type=receive_type, default=val)
        for key, val in parser.parse_args().__dict__.items():
            self.__setattr__(key, val)

        if self.test_only:
            assert self.model_path != '', 'You must give model_path on testing!'
        assert self.review_level in ['sentence', 'review'], '"review_level" must be equal to "sentence" or "review"!'

    def __str__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))
        to_str = ''
        for key, val in attributes:
            to_str += '{} = {}\n'.format(key, val)
        return to_str
