import os
import pickle
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from src.dataset import Dataset, batch_loader
from src.evaluate import evaluate_mse
from src.helpers import get_logger, date
from src.word2vec import Word2vec
from src.model import UMPR


def training(train_dataloader, valid_dataloader, model, config, model_path):
    logger.info('Start to train!')
    valid_mse = evaluate_mse(model, valid_dataloader)
    logger.info(f'Initial validation mse is {valid_mse:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.lr_decay)

    best_loss, batch_counter = 100, 0
    for epoch in range(config.train_epochs):
        total_loss, total_samples = 0, 0
        for batch in tqdm(train_dataloader, desc=f'Training epoch {epoch}', leave=False):
            model.train()
            pred, loss = model(*batch)
            loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(pred)
            total_samples += len(pred)

            batch_counter += 1
            if batch_counter % 500 == 0:
                valid_mse = evaluate_mse(model, valid_dataloader)
                logger.info(f'\rEpoch {epoch:2d}; batch {batch_counter:5d}; '
                            f'train loss {total_loss / total_samples:.6f}; valid mse {valid_mse:.6f}')
                if best_loss > valid_mse:
                    if hasattr(model, 'module'):
                        torch.save(model.module, model_path)
                    else:
                        torch.save(model, model_path)
                    best_loss = valid_mse

        lr_sch.step()
        logger.info(f'Epoch {epoch:3d} done; train loss {total_loss / total_samples:.6f}')
        if batch_counter > 50000:
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
        train_data = Dataset(train_path, photo_json, photo_path, w2v, config)
        logger.debug('Loading valid dataset.')
        valid_data = Dataset(valid_path, photo_json, photo_path, w2v, config)
        pickle.dump([train_data, valid_data], open(config.data_dir + '/dataset.pkl', 'wb'))

    logger.info(f'Training dataset contains {len(train_data.data[0])} samples.')
    train_dlr = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                           collate_fn=lambda x: batch_loader(x, config.review_net_only))
    valid_dlr = DataLoader(valid_data, batch_size=config.batch_size,
                           collate_fn=lambda x: batch_loader(x, config.review_net_only))

    if config.multi_gpu:
        model = torch.nn.DataParallel(UMPR(config, w2v.embedding)).to(config.device)
    else:
        model = UMPR(config, w2v.embedding).to(config.device)
    training(train_dlr, valid_dlr, model, config, config.model_path)


def test():
    logger.debug('Loading test dataset.')
    test_data = Dataset(test_path, photo_json, photo_path, w2v, config)
    test_dlr = DataLoader(test_data, batch_size=config.batch_size,
                          collate_fn=lambda x: batch_loader(x, config.review_net_only))
    logger.info('Start to test.')
    if config.multi_gpu:
        model = torch.nn.DataParallel(torch.load(config.model_path)).to(config.device)
    else:
        model = torch.load(config.model_path)
    test_loss = evaluate_mse(model, test_dlr)
    logger.info(f"Test end, test mse is {test_loss:.6f}")


if __name__ == '__main__':
    config = Config()

    if config.test_only:
        if not os.path.exists(config.model_path):
            print(f'{config.model_path} is not exist! Please train first (set test_only=False in config.py)!')
            exit(-1)
    else:
        save_name = os.path.basename(config.data_dir.strip("/")) + ('_review_net' if config.review_net_only else '')
        config.log_path = f'./log/{save_name}{date("%Y%m%d_%H%M%S")}.txt'
        config.model_path = f'./model/{save_name}{date("%Y%m%d_%H%M%S")}.pt'
        os.makedirs(os.path.dirname(config.log_path), exist_ok=True)
        os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
    photo_path = os.path.join(config.data_dir, 'photos')
    photo_json = os.path.join(config.data_dir, 'photos.json')
    train_path = os.path.join(config.data_dir, 'train.csv')
    valid_path = os.path.join(config.data_dir, 'valid.csv')
    test_path = os.path.join(config.data_dir, 'test.csv')

    logger = get_logger(config.log_path)
    logger.info(config)
    logger.info(f'Logging to {config.log_path}')
    logger.info(f'Save model {config.model_path}')
    logger.info(f'Photo path {photo_path}')
    logger.info(f'Photo json {photo_json}')
    logger.info(f'Train file {train_path}')
    logger.info(f'Valid file {valid_path}')
    logger.info(f'Test  file {test_path}\n')

    w2v = Word2vec(config.word2vec_file)

    if not config.test_only:
        train()
    test()
