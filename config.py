import torch


class Config:
    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    train_epochs = 20
    batch_size = 128
    learning_rate = 0.001
    l2_regularization = 1e-6
    learning_rate_decay = 0.99

    review_count = 10
    review_length = 30
    lowest_review_count = 2
    PAD_WORD = '<UNK>'

    gru_hidden_size = 64  # R-net. It's u in paper
    self_attention_hidden_size = 64  # S-net. It's us in paper
