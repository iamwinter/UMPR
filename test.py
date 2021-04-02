import os
import torch
from torch.utils.data import DataLoader

from config import Config
from src import Dataset, predict_mse, load_embedding, batch_loader, process_bar, get_logger


if __name__ == '__main__':
    config = Config()

    photo_path = os.path.join(config.data_dir, 'photos')
    photo_json = os.path.join(config.data_dir, 'photos.json')
    test_path = os.path.join(config.data_dir, 'test.csv')

    logger = get_logger()
    logger.info(config)
    logger.info(f'Trained model {config.model_path}')
    logger.info(f'Photo path {photo_path}')
    logger.info(f'Photo json {photo_json}')
    logger.info(f'Test  file {test_path}\n')

    if not os.path.exists(config.model_path):
        print(f'{config.model_path} is not exist! Please train first!')
        exit(-1)

    logger.debug('Load word embedding...')
    word_emb, word_dict = load_embedding(config.word2vec_file)

    logger.debug('Loading test dataset.')
    test_data = Dataset(test_path, photo_json, word_dict, config)
    test_dlr = DataLoader(test_data, batch_size=config.batch_size, collate_fn=lambda x: batch_loader(x, photo_path))

    model = torch.load(config.model_path).to(config.device)
    test_loss = predict_mse(model, test_dlr)
    logger.info(f"Test end, test mse is {test_loss:.6f}")
