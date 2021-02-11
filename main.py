import os
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Config
from utils import Dataset, date, predict_mse, load_embedding
from model import UMPR


def train(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start up the training!')
    train_mse = predict_mse(model, train_dataloader)
    valid_mse = predict_mse(model, valid_dataloader)
    print(f'{date()}#### Initial train mse {train_mse:.6f}, validation mse {valid_mse:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)

    best_loss, best_epoch = 100, 0
    for epoch in range(config.train_epochs):
        model.train()
        total_loss, total_samples = 0, 0
        for batch in train_dataloader:
            user_reviews, item_reviews, reviews, ratings = map(lambda x: x.to(config.device), batch)
            predict = model(user_reviews, item_reviews, reviews)
            loss = F.mse_loss(predict, ratings, reduction='mean')
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(predict)
            total_samples += len(predict)

        lr_sch.step()
        model.eval()
        valid_mse = predict_mse(model, valid_dataloader)
        train_loss = total_loss / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; validation mse {valid_mse:.6f}")

        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')


if __name__ == '__main__':
    config = Config()
    print(config)
    print(f'{date()}## Load word embedding and dataset...')
    word_emb, word_dict = load_embedding(config.word2vec_file)

    train_dataset = Dataset(config.train_file, word_dict, config)
    valid_dataset = Dataset(config.valid_file, word_dict, config)
    test_dataset = Dataset(config.test_file, word_dict, config)
    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size)

    Model = UMPR(config, word_emb).to(config.device)
    del word_emb, word_dict, train_dataset, valid_dataset, test_dataset

    if '/' in config.saved_model:
        os.makedirs(os.path.dirname(config.saved_model), exist_ok=True)  # mkdir if not exist
    train(train_dlr, valid_dlr, Model, config, config.saved_model)

    test_loss = predict_mse(torch.load(config.saved_model), test_dlr)
    print(f"{date()}## Test end, test mse is {test_loss:.6f}")
