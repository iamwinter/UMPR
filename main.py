import os
import time
import torch
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
            predict, loss = model(user_reviews, item_reviews, reviews, photos, ratings)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(ratings)
            total_samples += len(ratings)
            process_bar(i + 1, len(train_dataloader), prefix=' Training ')

        lr_sch.step()
        model.eval()
        valid_mse = predict_mse(model, valid_dataloader)
        train_loss = total_loss / total_samples
        logger.info(f"## Epoch {epoch:3d}; train mse {train_loss:.6f}; validation mse {valid_mse:.6f}")

        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)

    end_time = time.perf_counter()
    second = int(end_time - start_time)
    logger.info(f'End of training! Time used {second / 60}:{second % 60}.')


if __name__ == '__main__':
    config = Config()

    log_path = f'./log/{date("%Y%m%d_%H%M%S")}.txt'
    model_path = f'./model/{date("%Y%m%d_%H%M%S")}.pt'
    photo_path = os.path.join(config.data_dir, 'photos')
    photo_json = os.path.join(config.data_dir, 'photos.json')
    train_path = os.path.join(config.data_dir, 'train.csv')
    valid_path = os.path.join(config.data_dir, 'valid.csv')
    test_path = os.path.join(config.data_dir, 'test.csv')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    logger = get_logger(log_path)
    logger.info(config)
    logger.info(f'Logging to {log_path}')
    logger.info(f'Save model {model_path}')
    logger.info(f'Photo path {photo_path}')
    logger.info(f'Photo json {photo_json}')
    logger.info(f'Train file {train_path}')
    logger.info(f'Valid file {valid_path}')
    logger.info(f'Test  file {test_path}\n')

    logger.debug('Load word embedding...')
    word_emb, word_dict = load_embedding(config.word2vec_file)

    logger.debug('Load all of photos')
    photos_dict = load_photos(photo_path)
    logger.info(f'Loaded {len(photos_dict)} intact photos.')

    logger.debug('Loading train dataset.')
    train_dataset = Dataset(train_path, photo_json, word_dict, config)
    logger.debug('Loading valid dataset.')
    valid_dataset = Dataset(valid_path, photo_json, word_dict, config)
    logger.debug('Loading test dataset.')
    test_dataset = Dataset(test_path, photo_json, word_dict, config)

    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                           collate_fn=lambda x: batch_loader(x, config, photos_dict))
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size,
                           collate_fn=lambda x: batch_loader(x, config, photos_dict))
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size,
                          collate_fn=lambda x: batch_loader(x, config, photos_dict))

    Model = UMPR(config, word_emb).to(config.device)
    train(train_dlr, valid_dlr, Model, config, model_path)

    test_loss = predict_mse(torch.load(model_path), test_dlr)
    logger.info(f"Test end, test mse is {test_loss:.6f}")
