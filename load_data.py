import json

import torch

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, pipeline

import random
import jieba

jieba.setLogLevel(jieba.logging.ERROR)


class ECGDataset(Dataset):
    def __init__(self, path, for_transformer=False):
        data = open(path, "r")
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
        self.for_transformer = for_transformer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Data format: [text, label]
        if not self.for_transformer:
            sentence = self.tokenizer(self.samples[idx][0], return_tensors="pt")["input_ids"]
            label = torch.tensor(self.samples[idx][1], dtype=torch.long)
            return {
                "input_ids": sentence.squeeze(0),
                "label": label,
                "text": self.samples[idx][0],
            }
        else:
            sentence = self.tokenizer(
                self.samples[idx][0],
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            label = torch.tensor(self.samples[idx][1], dtype=torch.long)
            return {
                "input_ids": sentence["input_ids"].squeeze(0),
                "attention_mask": sentence["attention_mask"].squeeze(0),
                "label": label,
                "text": self.samples[idx][0],
            }



class SMP2020Dataset(Dataset):
    def __init__(self, path, do_augment=False):
        # 加载数据
        data = json.load(open(path, "r", encoding="utf-8"))
        samples = []
        self.emotions = ["neutral", "happy", "angry", "sad", "fear", "surprise"]
        for sample in tqdm(data, desc="Loading data"):
            samples.append([sample["content"], self.emotions.index(sample["label"])])

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.vocab_size = self.tokenizer.vocab_size
        print(f" * Load data from {path} with {len(samples)} samples")
        print(f" * Tokenizer: bert-base-chinese with vocab size {self.vocab_size}")
        print(f" * Pad token id: {self.tokenizer.pad_token_id}")
        self.samples = samples
        self.do_augment = do_augment

        if self.do_augment:
            # 仅在初始化时加载一次 fill-mask pipeline，
            # 为避免 CUDA 在子进程中重新初始化问题，这里强制使用 CPU（device=-1）
            self.fill_mask = pipeline("fill-mask", model="bert-base-chinese", device=-1)

    def __len__(self):
        return len(self.samples)

    def _bert_synonym_replacement(self, sentence, n=5):
        """
        使用预训练 BERT 模型进行近义词替换：
        - 随机选择句子中的 n 个词，将其替换为 [MASK]
        - 通过模型预测候选词，并替换原词（选择与原词不同的第一个预测结果）
        """
        words = list(jieba.cut(sentence))
        new_words = words.copy()
        indices = list(range(len(words)))
        random.shuffle(indices)
        replaced = 0

        for idx in indices:
            # 构造 mask 后的句子。中文中可直接拼接，无需额外空格
            masked_sentence = (
                "".join(new_words[:idx]) + "[MASK]" + "".join(new_words[idx + 1 :])
            )
            predictions = self.fill_mask(masked_sentence, top_k=5)

            # 从预测结果中选取与原词不同的候选词进行替换
            for pred in predictions:
                token = pred["token_str"].strip()
                if token != new_words[idx]:
                    new_words[idx] = token
                    replaced += 1
                    break
            if replaced >= n:
                break
        return "".join(new_words)

    def _random_deletion(self, sentence, p=0.2):
        """
        随机删除：以概率 p 删除句子中的词语。如果删除后为空，则随机保留一个词。
        """
        words = list(jieba.cut(sentence))
        if len(words) <= 1:
            return sentence

        new_words = []
        for word in words:
            if random.uniform(0, 1) > p:
                new_words.append(word)
        if not new_words:
            new_words.append(random.choice(words))
        return "".join(new_words)

    def _random_swap(self, sentence, n=1):
        """
        随机交换：随机交换句子中两个词的位置，交换 n 次。
        """
        words = list(jieba.cut(sentence))
        new_words = words.copy()
        length = len(new_words)
        if length < 2:
            return sentence

        for _ in range(n):
            idx1, idx2 = random.sample(range(length), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        return "".join(new_words)

    def augment_text(self, sentence):
        """
        随机选择一种数据增强方法：
          - "bert_synonym"：基于 BERT 的近义词替换
          - "delete"     ：随机删除
          - "swap"       ：随机交换
        """
        method = random.choice(["bert_synonym", "delete", "swap"])
        if method == "bert_synonym":
            return self._bert_synonym_replacement(sentence, n=5)
        elif method == "delete":
            return self._random_deletion(sentence, p=0.5)
        elif method == "swap":
            return self._random_swap(sentence, n=5)
        else:
            return sentence

    def __getitem__(self, idx):
        original_sentence = self.samples[idx][0]
        sentence = original_sentence

        if self.do_augment and random.random() < 0.6:
            sentence = self.augment_text(sentence)

        input_ids = self.tokenizer(sentence, return_tensors="pt")["input_ids"]
        label = torch.tensor(self.samples[idx][1], dtype=torch.long)

        return {
            "input_ids": input_ids.squeeze(0),
            "label": label,
            "text": sentence,
            "original_text": original_sentence,
            "auged": sentence != original_sentence,
        }


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["label"] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return {"input_ids": input_ids, "label": labels}


if __name__ == "__main__":
    # data = json.load(open("/mnt/d/codes/Swc_Data/smp2020/train/usual_train.txt", "r"))
    # emotions = ["neutral", "happy", "angry", "sad", "fear", "surprise"]
    # for sample in data:
    #     print(sample["content"], sample["label"], emotions.index(sample["label"]))

    train_dataset = SMP2020Dataset(
        "/mnt/d/codes/Swc_Data/smp2020/train/usual_train.txt", do_augment=True
    )
    for idx, sample in enumerate(train_dataset):
        if sample["auged"]:
            print(sample["text"], sample["original_text"], sample["auged"])
        if idx > 100:
            break
