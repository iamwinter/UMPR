import os
import pickle
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Config
from model import Dataset, UMPR


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


def mse_loss(model, dataloader):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_reviews, item_reviews, reviews, ratings = map(lambda x: x.to(config.device), batch)
            predict = model(user_reviews, item_reviews, reviews)
            mse += F.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count


def train(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start the training!')
    train_mse = mse_loss(model, train_dataloader)
    valid_mse = mse_loss(model, valid_dataloader)
    print(f'{date()}#### Initial train mse {train_mse:.6f}, validation mse {valid_mse:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)

    best_loss, best_epoch = 100, 0
    for epoch in range(config.train_epochs):
        model.train()
        total_loss, total_samples = 0, 0
        for batch in train_dataloader:
            user_reviews, item_reviews, reviews, ratings = map(lambda x: x.to(config.device), batch)
            predict = model(user_reviews, item_reviews, reviews)
            loss = F.mse_loss(predict, ratings, reduction='sum')
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_samples += len(predict)

        model.eval()
        valid_mse = mse_loss(model, valid_dataloader)
        train_loss = total_loss / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; validation mse {valid_mse:.6f}")

        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')


def test(dataloader, best_model):
    print(f'{date()}## Start the testing!')
    start_time = time.perf_counter()
    test_loss = mse_loss(best_model, dataloader)
    end_time = time.perf_counter()
    print(f"{date()}## Test end, test mse is {test_loss:.6f}, time used {end_time - start_time:.0f} seconds.")


if __name__ == '__main__':
    config = Config()
    print(f'{date()}## Load embedding and data...')
    word_emb = pickle.load(open('data/embedding/word_emb.pkl', 'rb'), encoding='iso-8859-1')
    word_dict = pickle.load(open('data/embedding/dict.pkl', 'rb'), encoding='iso-8859-1')

    train_dataset = Dataset('data/music/train.csv', word_dict, config)
    valid_dataset = Dataset('data/music/valid.csv', word_dict, config)
    test_dataset = Dataset('data/music/test.csv', word_dict, config)
    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size)

    Model = UMPR(config, word_emb).to(config.device)
    del word_emb, word_dict, train_dataset, valid_dataset, test_dataset

    os.makedirs('model', exist_ok=True)  # mkdir if not exist
    model_Path = 'model/best_model.pt'
    train(train_dlr, valid_dlr, Model, config, model_Path)
    test(test_dlr, torch.load(model_Path))
