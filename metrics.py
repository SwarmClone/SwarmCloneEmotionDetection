import os
import time
import torch

from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy

def cal_metrics(model, dataloader, num_classes, is_train=False):
    torch.cuda.empty_cache()
    model.eval()
    
    acc = MulticlassAccuracy(num_classes=num_classes, average='macro').to(model.device)

    i = 1
    train_part = len(dataloader) // 10
    
    for batch in tqdm(dataloader, desc=f"Metrics Progress for {'Train' if is_train else 'Val'}", unit="batch"):
        x = batch["input_ids"].to(model.device)
        y = batch["label"].to(model.device)
        y_hat, _ = model(x)
        acc.update(y_hat, y)
        del x, y, y_hat

        i += 1
        if is_train and i > train_part:
            break
        
    avg_acc = acc.compute().item()

    print(f" * Accuracy: {avg_acc} \n")
    time.sleep(1)
    os.system("clear")
    model.train()
    return avg_acc
    
