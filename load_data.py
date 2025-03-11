import json

import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, path):
        data = open(path, 'r')
        data = json.load(data)
        samples = []
        for sample in data:
            sample[0][0] = sample[0][0].replace("\n", "").replace(" ", "")
            sample[1][0] = sample[1][0].replace("\n", "").replace(" ", "")
            samples.append(sample[0])
            samples.append(sample[1])
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.vocab_size = self.tokenizer.vocab_size
        print(f" * Load data from {path} with {len(samples)} samples")
        print(f" * Tokenizer: bert-base-chinese with vocab size {self.vocab_size}")
        print(f" * Pad token id: {self.tokenizer.pad_token_id}")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Data format: [text, label]
        sentence = self.tokenizer(self.samples[idx][0], return_tensors="pt")["input_ids"]
        label = torch.tensor(self.samples[idx][1], dtype=torch.long)
        return {"input_ids": sentence.squeeze(0), "label": label}


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["label"] for item in batch]
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.stack(labels)    
    return {"input_ids": input_ids, "label": labels}


if __name__ == "__main__":
    train_ds = TextDataset("/mnt/d/codes/Swc_Data/ecg_data/ecg_train_data.json")
    train_iter = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=19, collate_fn=collate_fn)
    
    for i in train_iter:
        print(i["input_ids"].shape)
        print(i["label"].shape)
        print(i["input_ids"])
        print(i["label"])
        break
    
    