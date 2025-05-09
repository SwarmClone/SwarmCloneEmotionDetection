import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf
from torch import Generator
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import seed_everything

from models.bilstm import PlBiLSTM
from models.transformer import PlTransformer
from load_data import ECGDataset, SMP2020Dataset, collate_fn
from metrics import cal_metrics


class PlTextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path,
        val_ratio=0.1,
        batch_size=32,
        num_workers=19,
        used_dataset="SMP2020",
        for_transformer=False,
        max_len=256,
        do_augment=False,
    ):
        super().__init__()
        self.used_dataset = used_dataset
        if used_dataset == "ECG":
            self.dataset = ECGDataset(path, for_transformer, max_len)
        elif used_dataset == "SMP2020":
            assert OmegaConf.is_list(path), "SMP2020 dataset need train and test path"
            self.dataset = None
            self.train_dataset = SMP2020Dataset(path[0], for_transformer, max_len, do_augment=do_augment)
            self.val_dataset = SMP2020Dataset(path[1], for_transformer, max_len, do_augment=do_augment)
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.seed = 42
        self.num_workers = num_workers
        self.for_transformer = for_transformer

    def setup(self, stage=None):
        if self.used_dataset == "ECG":
            num_of_data = len(self.dataset)
            num_of_val = int(num_of_data * self.val_ratio)
            num_of_train = num_of_data - num_of_val

            self.train_dataset, self.val_dataset = random_split(
                self.dataset,
                [num_of_train, num_of_val],
                generator=Generator().manual_seed(self.seed),
            )
        elif self.used_dataset == "SMP2020":
            pass
        else:
            raise ValueError(f"Dataset {self.used_dataset} not found")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn if not self.for_transformer else None,
            pin_memory=True,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn if not self.for_transformer else None,
            pin_memory=True,
            prefetch_factor=2,
        )


class MetricsCallback(pl.Callback):
    def __init__(self, train_dataloader, val_dataloader):
        super().__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def on_train_epoch_end(self, trainer, pl_module):
        avg_acc = cal_metrics(
            pl_module,
            self.train_dataloader,
            num_classes=6,
            for_transformer=False,
            is_train=True,
        )
        pl_module.log("metrics/train_acc", avg_acc, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        avg_acc = cal_metrics(pl_module, self.val_dataloader, num_classes=6, for_transformer=False)
        pl_module.log("metrics/val_acc", avg_acc, prog_bar=True, on_epoch=True)


if __name__ == "__main__":
    seed_everything(42)
    torch.set_float32_matmul_precision("high")

    config = OmegaConf.load("configs/bilstm.yaml")

    model = PlBiLSTM(**config.model.params)
    # model = PlTransformer(**config.model.params)
    data = PlTextDataModule(**config.data.params)
    data.setup()

    logger = pl.loggers.TensorBoardLogger(**config.logger.params)

    callbacks = []
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch}-{metrics/val_acc:.4f}",
        monitor="metrics/val_acc",
        auto_insert_metric_name=False,
        mode="max",
        save_top_k=3,
        
    )

    metrics_callback = MetricsCallback(data.train_dataloader(), data.val_dataloader())
    callbacks.extend([ckpt_callback, metrics_callback])
    trainer = pl.Trainer(**config.lightning.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(model, data)
