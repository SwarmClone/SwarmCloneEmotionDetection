import torch
import pytorch_lightning as pl

from torch import nn
from timm.loss import LabelSmoothingCrossEntropy


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_classes,
        num_heads=8,
        use_extra_mlp=True,
        extra_mlp_ratio=4,
        mlp_ratio=4,
        dropout=0.2,
        embedding_dropout=0.2,
        max_len=512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embedding_dim))
        self.embedding_dropout = (
            nn.Dropout(p=embedding_dropout) if embedding_dropout > 0 else nn.Identity()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        if use_extra_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * extra_mlp_ratio),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(embedding_dim * extra_mlp_ratio, num_classes),
            )
        else:
            self.mlp = nn.Linear(embedding_dim, num_classes)

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, x, mask):
        x = self.embedding(x)
        x = x + self.positional_encoding
        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        logits = self.mlp(x[:, -1, :])
        out = self.softmax(logits)
        return logits, out

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class PiTransformer(pl.LightningModule):
    def __init(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_classes,
        num_heads=8,
        use_extra_mlp=True,
        extra_mlp_ratio=4,
        mlp_ratio=4,
        dropout=0.2,
        embedding_dropout=0.2,
        max_len=512,
        learning_rate=1e-3,
        weight_decay=1e-4,
        label_smoothing=0.1,
    ):
        super().__init__()
        self.model = TransformerModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            num_heads=num_heads,
            use_extra_mlp=use_extra_mlp,
            extra_mlp_ratio=extra_mlp_ratio,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            embedding_dropout=embedding_dropout,
            max_len=max_len,
        )
        self.train_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.val_loss = nn.CrossEntropyLoss()
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x, mask):
        return self.model(x, mask)

    def training_step(self, batch, batch_idx):
        x = batch["input_ids"]
        mask = batch["attention_mask"].bool()
        y = batch["label"]
        logits, _ = self.model(x, mask)
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
        mask = batch["attention_mask"].bool()
        y = batch["label"]
        logits, _ = self.model(x, mask)
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


if __name__ == "__main__":
    from load_data import ECGDataset
    from torch.utils.data import DataLoader
    from torchinfo import summary

    train_ds = ECGDataset("/mnt/d/codes/Swc_Data/ecg_data/ecg_train_data.json", for_transformer=True)
    train_iter = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=19)

    model = TransformerModel(
        vocab_size=21128,
        embedding_dim=512,
        hidden_dim=512,
        num_layers=6,
        num_classes=6,
        num_heads=8,
        use_extra_mlp=False,
        mlp_ratio=4,
        dropout=0.2,
        embedding_dropout=0,
        max_len=256,
    )
    print(summary(model))
    
    for i in train_iter:
        x = i["input_ids"]
        mask = i["attention_mask"].bool()
        logits, out = model(x, mask)
        print(logits)
        print(out)
        break
