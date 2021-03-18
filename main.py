import os
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Config
from utils import Dataset, date, predict_mse, load_embedding, batch_loader, load_photos, process_bar, get_logger
from model import UMPR


def train(train_dataloader, valid_dataloader, model, config, model_path):
    logger.info('Start to train!')
    train_mse = predict_mse(model, train_dataloader)
    valid_mse = predict_mse(model, valid_dataloader)
    logger.info(f'Initial train mse {train_mse:.6f}, validation mse {valid_mse:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)

    best_loss = 100
    for epoch in range(config.train_epochs):
        model.train()
        total_loss, total_samples = 0, 0
        for i, batch in enumerate(train_dataloader):
            user_reviews, item_reviews, reviews, photos, ratings = map(lambda x: x.to(config.device), batch)
            predict = model(user_reviews, item_reviews, reviews, photos)
            loss = F.mse_loss(predict, ratings, reduction='mean')
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(predict)
            total_samples += len(predict)
            process_bar(i, len(train_dataloader), prefix=' Training ')

        lr_sch.step()
        model.eval()
        valid_mse = predict_mse(model, valid_dataloader)
        train_loss = total_loss / total_samples
        logger.info(f"## Epoch {epoch:3d}; train mse {train_loss:.6f}; validation mse {valid_mse:.6f}")

        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)

    end_time = time.perf_counter()
    logger.info(f'End of training! Time used {end_time - start_time:.0f} seconds.')


if __name__ == '__main__':
    config = Config()

    os.makedirs('./log', exist_ok=True)
    logger = get_logger(f'./log/{date("%Y%m%d_%H%M%S")}.txt')
    logger.info(config)

    logger.debug('Load word embedding...')
    word_emb, word_dict = load_embedding(config.word2vec_file)

    logger.debug('Load all of photos')
    photos_dict = load_photos(os.path.join(config.data_dir, 'photos'))
    logger.info(f'Loaded {len(photos_dict)} intact photos.')

    logger.debug('Loading train dataset.')
    train_dataset = Dataset(os.path.join(config.data_dir, 'train.csv'), word_dict, config)
    logger.debug('Loading valid dataset.')
    valid_dataset = Dataset(os.path.join(config.data_dir, 'valid.csv'), word_dict, config)
    logger.debug('Loading test dataset.')
    test_dataset = Dataset(os.path.join(config.data_dir, 'test.csv'), word_dict, config)

    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                           collate_fn=lambda x: batch_loader(x, config, photos_dict))
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size,
                           collate_fn=lambda x: batch_loader(x, config, photos_dict))
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size,
                          collate_fn=lambda x: batch_loader(x, config, photos_dict))

    Model = UMPR(config, word_emb).to(config.device)
    if '/' in config.saved_model:
        os.makedirs(os.path.dirname(config.saved_model), exist_ok=True)  # mkdir if not exist
    train(train_dlr, valid_dlr, Model, config, config.saved_model)

    test_loss = predict_mse(torch.load(config.saved_model), test_dlr)
    logger.info(f"Test end, test mse is {test_loss:.6f}")
