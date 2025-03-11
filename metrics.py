import os
import time

from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy

def cal_metrics(model, dataloader, num_classes):
    model.eval()
    
    acc = MulticlassAccuracy(num_classes=num_classes, average='macro').to(model.device)
    for batch in tqdm(dataloader, desc="Metrics Progress", unit="batch"):
        x = batch["input_ids"].to(model.device)
        y = batch["label"].to(model.device)
        y_hat, _ = model(x)
        acc.update(y_hat, y)
        
    avg_acc = acc.compute().item()

    print(f" * Accuracy: {avg_acc} \n")
    time.sleep(1)
    os.system("clear")
    return avg_acc
    
