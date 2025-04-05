import math
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
        # self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embedding_dim))
        self.pos_encoding = self.positional_encoding(max_len, embedding_dim)
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
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        if use_extra_mlp:
            self.mlp = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(embedding_dim, embedding_dim * extra_mlp_ratio),
                nn.LayerNorm(embedding_dim * extra_mlp_ratio),
                nn.GELU(),
                nn.Linear(embedding_dim * extra_mlp_ratio, num_classes),
            )
        else:
            self.mlp = nn.Linear(embedding_dim, num_classes)

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, x, mask):
        x = self.embedding(x)
        # x = x + self.positional_encoding
        x = x + self.pos_encoding[: x.size(1), :]
        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        # logits = self.mlp(x[:, 0, :])
        logits = self.mlp(x.mean(dim=1))
        out = self.softmax(logits)
        return logits, out

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.constant_(p, 0)
        
        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1)


    def positional_encoding(self, max_len, embedding_dim):
        """
        Generate cosine positional encoding matrix with shape [max_len, embedding_dim].
        """
        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim)
        )

        pos_enc = torch.zeros(max_len, embedding_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        return pos_enc.unsqueeze(0).cuda()


class PlTransformer(pl.LightningModule):
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
        lr=1e-3,
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
        # self.train_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.train_loss = nn.CrossEntropyLoss()
        self.val_loss = nn.CrossEntropyLoss()
        self.lr = lr
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
        try:
            print(torch.max(self.model.transformer_encoder.layers[0].linear2.weight.grad), torch.min(self.model.transformer_encoder.layers[0].linear2.weight.grad))
            print(torch.max(self.model.transformer_encoder.layers[-1].linear2.weight.grad), torch.min(self.model.transformer_encoder.layers[-1].linear2.weight.grad))
        except: 
            pass
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=0
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
    from load_data import ECGDataset
    from torch.utils.data import DataLoader
    from torchinfo import summary

    train_ds = ECGDataset(
        "/mnt/d/codes/Swc_Data/ecg_data/ecg_train_data.json", for_transformer=True
    )
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

    for idx, i in enumerate(train_iter):
        x = i["input_ids"]
        mask = i["attention_mask"].bool()
        print(x)
        # logits, out = model(x, mask)
        # print(logits)
        # print(out)

        if idx > 3:
            break
