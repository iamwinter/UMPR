import torch
from torch.nn import functional as F
from tqdm import tqdm


def evaluate_mse(model, dataloader):
    mse, sample_count = 0, 0
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader, desc='Evaluate', leave=False):
            pred, loss = model(*batch)
            mse += F.mse_loss(pred, batch[-1].to(pred.device), reduction='sum').item()
            sample_count += len(pred)
    return mse / sample_count
