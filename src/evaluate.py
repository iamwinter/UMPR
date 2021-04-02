import torch
from torch.nn import functional as F
from src.helpers import process_bar


def predict_mse(model, dataloader):
    mse, sample_count = 0, 0
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            process_bar(i + 1, len(dataloader), prefix='Evaluate')
            pred, loss = model(*batch)
            mse += F.mse_loss(pred, batch[-1].to(pred.device), reduction='sum').item()
            sample_count += len(pred)
    return mse / sample_count
