import os
import pickle
import time
import torch
from torch.utils.data import DataLoader

from config import Config
from utils import Dataset, date, predict_mse, load_embedding, batch_loader, process_bar, get_logger
from model import UMPR


def training(train_dataloader, valid_dataloader, model, config, model_path):
    logger.info('Start to train!')
    valid_mse = predict_mse(model, valid_dataloader)
    logger.info(f'Initial validation mse is {valid_mse:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)

    best_loss, batch_counter = 100, 0
    for epoch in range(config.train_epochs):
        total_loss, total_samples = 0, 0
        for i, batch in enumerate(train_dataloader):
            process_bar(i + 1, len(train_dataloader), prefix=f'Training epoch {epoch}')
            model.train()
            pred, loss = model(*batch)
            loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(pred)
            total_samples += len(pred)

            batch_counter += 1
            if batch_counter % 500 == 0 or batch_counter % 500 > 100 and i + 1 == len(train_dataloader):
                valid_mse = predict_mse(model, valid_dataloader)
                logger.info(f'## Evaluate model on validation and loss is {valid_mse:.6f}')
                if best_loss > valid_mse:
                    if hasattr(model, 'module'):
                        torch.save(model.module, model_path)
                    else:
                        torch.save(model, model_path)
                    best_loss = valid_mse

        lr_sch.step()
        train_loss = total_loss / total_samples
        logger.info(f"Epoch {epoch:3d}; train loss {train_loss:.6f}")
        if batch_counter > 20000:
            break

    end_time = time.perf_counter()
    second = int(end_time - start_time)
    logger.info(f'End of training! Time used {second // 3600}:{second % 3600 // 60}:{second % 60}.')


def train():
    try:
        train_data, valid_data = pickle.load(open(config.data_dir + '/dataset.pkl', 'rb'))
        logger.info('Loaded dataset from dataset.pkl!')
    except Exception:
        logger.debug('Loading train dataset.')
        train_data = Dataset(train_path, photo_json, photo_path, word_dict, config)
        logger.debug('Loading valid dataset.')
        valid_data = Dataset(valid_path, photo_json, photo_path, word_dict, config)
        pickle.dump([train_data, valid_data], open(config.data_dir + '/dataset.pkl', 'wb'))

    train_dlr = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=2,
                           collate_fn=lambda x: batch_loader(x))
    valid_dlr = DataLoader(valid_data, batch_size=config.batch_size, collate_fn=lambda x: batch_loader(x))

    # model = UMPR(config, word_emb).to(config.device)
    model = torch.nn.DataParallel(UMPR(config, word_emb)).to(config.device)
    training(train_dlr, valid_dlr, model, config, config.model_path)


def test():
    logger.debug('Loading test dataset.')
    test_data = Dataset(test_path, photo_json, photo_path, word_dict, config)
    test_dlr = DataLoader(test_data, batch_size=config.batch_size * 2, collate_fn=lambda x: batch_loader(x))
    logger.info('Start to test.')
    model = torch.load(config.model_path)
    test_loss = predict_mse(model, test_dlr)
    logger.info(f"Test end, test mse is {test_loss:.6f}")


if __name__ == '__main__':
    config = Config()

    config.log_path = f'./log/{os.path.basename(config.data_dir.strip("/"))}{date("%Y%m%d_%H%M%S")}.txt'
    config.model_path = f'./model/{os.path.basename(config.data_dir.strip("/"))}{date("%Y%m%d_%H%M%S")}.pt'
    photo_path = os.path.join(config.data_dir, 'photos')
    photo_json = os.path.join(config.data_dir, 'photos.json')
    train_path = os.path.join(config.data_dir, 'train.csv')
    valid_path = os.path.join(config.data_dir, 'valid.csv')
    test_path = os.path.join(config.data_dir, 'test.csv')
    os.makedirs(os.path.dirname(config.log_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)

    logger = get_logger(config.log_path)
    logger.info(config)
    logger.info(f'Logging to {config.log_path}')
    logger.info(f'Save model {config.model_path}')
    logger.info(f'Photo path {photo_path}')
    logger.info(f'Photo json {photo_json}')
    logger.info(f'Train file {train_path}')
    logger.info(f'Valid file {valid_path}')
    logger.info(f'Test  file {test_path}\n')

    logger.debug('Load word embedding...')
    word_emb, word_dict = load_embedding(config.word2vec_file)

    train()
    test()
