import torch
import pytorch_lightning as pl

from torch import nn
from timm.loss import LabelSmoothingCrossEntropy

class BiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_classes,
        use_extra_mlp=True,
        mlp_ratio=[4, 2],
        mlp_dropout=[0.5, 0.2],
        recurrent_dropout=0.2,
        embedding_dropout=0.2
    ):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_dropout > 0:
            self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        else:
            self.embedding_dropout = nn.Identity()
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=recurrent_dropout,
        )
        self.use_extra_mlp = use_extra_mlp
        if self.use_extra_mlp:
            print(" * Using extra mlp with ratio: ", mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * mlp_ratio[0]),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout[0]),
                nn.Linear(hidden_dim * mlp_ratio[0], hidden_dim * mlp_ratio[1]),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout[1]),
                nn.Linear(hidden_dim * mlp_ratio[1], num_classes),
            )
        else:
            self.mlp = nn.Linear(hidden_dim * 2, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)
            
        out, _ = self.lstm(x)
        logits = self.mlp(out[:, -1, :])
        out = self.softmax(logits)
        return logits, out

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n // 4) : (n // 2)].fill_(1.0)
        if self.use_extra_mlp:
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
        else:
            nn.init.xavier_uniform_(self.mlp.weight)
            nn.init.constant_(self.mlp.bias, 0)


class PlBiLSTM(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_classes,
        use_extra_mlp=True,
        mlp_ratio=[4, 2],
        mlp_dropout=[0.5, 0.2],
        recurrent_dropout=0.2,
        embedding_dropout=0.2,
        lr=0.0001,
        weight_decay=0.01,
        label_smoothing=0.1
    ):
        super().__init__()
        self.bilstm = BiLSTM(
            vocab_size,
            embedding_dim,
            hidden_dim,
            num_layers,
            num_classes,
            use_extra_mlp,
            mlp_ratio,
            mlp_dropout,
            recurrent_dropout,
            embedding_dropout
        )
        self.train_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.val_loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        logits, out = self.bilstm(x)
        return logits, out

    def training_step(self, batch, batch_idx):
        x = batch["input_ids"]
        y = batch["label"]
        logits, _ = self(x)
        loss = self.train_loss(logits, y)

        self.log(
            "loss/train_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["input_ids"]
        y = batch["label"]
        logits, _ = self(x)
        loss = self.val_loss(logits, y)

        self.log(
            "loss/val_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return
    
    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=0
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # 按epoch更新
                "frequency": 1,  # 每个epoch更新一次
            },
        }


if __name__ == "__main__":
    
    from torch.utils.data import DataLoader
    from load_data import ECGDataset, collate_fn

    train_ds = ECGDataset("/mnt/d/codes/Swc_Data/ecg_data/ecg_train_data.json")
    train_iter = DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=19, collate_fn=collate_fn
    )

    vocab_size = train_ds.vocab_size
    embedding_dim = 512
    hidden_dim = 256
    num_layers = 2
    num_classes = 6

    model = BiLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, num_classes)
    for i in train_iter:
        x = i["input_ids"]
        y = i["label"]
        logits, out = model(x)
        print(logits, y)
        print(nn.CrossEntropyLoss()(logits, y))
        break
