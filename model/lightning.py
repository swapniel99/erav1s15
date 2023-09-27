import os
from torch import nn, optim
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from torchmetrics import MeanMetric

import utils
from .vanilla import Transformer
from dataset import RawDataset, BilingualDataset


class Model(LightningModule):
    def __init__(self, src_lang: str = 'en', tgt_lang: str = 'it', label_smoothing: float = 0.1,
                 batch_size: int = 64, learning_rate: float = 1e-3, enable_gc='batch') -> None:
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.transformer = None
        self.criterion = None
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.train_ds = None
        self.val_ds = None
        self.label_smoothing = label_smoothing
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.enable_gc = enable_gc
        self.my_train_loss = MeanMetric()
        self.my_val_loss = MeanMetric()

    def encode(self, *args):
        return self.transformer.encode(*args)

    def decode(self, *args):
        return self.transformer.decode(*args)

    def project(self, *args):
        return self.transformer.project(*args)

    def forward(self, *args):
        return self.transformer.forward(*args)

    def common_forward(self, batch):
        return self.forward(batch['encoder_input'],  # (B, seq_len)
                            batch['encoder_mask'],  # (B, 1, 1, seq_len)
                            batch['decoder_input'],  # (B, seq_len)
                            batch['decoder_mask']  # (B, 1, seq_len, seq_len)
                            )  # (B, seq_len, vocab_size)

    def common_step(self, batch):
        output = self.common_forward(batch)  # (B, seq_len, tgt_vocab_size)
        label = batch['label']  # (B, seq_len)
        loss = self.criterion(output.transpose(1, 2), label)
        del output, label
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.my_train_loss.update(loss, batch['label'].shape[0])
        self.log(f"train_loss", self.my_train_loss.compute(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.my_val_loss.update(loss, batch['label'].shape[0])
        self.log(f"val_loss", self.my_val_loss.compute(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.common_forward(batch)

    def prepare_data(self) -> None:
        RawDataset('opus_books', self.src_lang, self.tgt_lang)

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            rd = RawDataset('opus_books', self.src_lang, self.tgt_lang)
            train_ds_raw, val_ds_raw = rd.split(0.9)
            self.train_ds = BilingualDataset(train_ds_raw, self.src_lang, self.tgt_lang, rd.src_tokenizer,
                                             rd.tgt_tokenizer, batch_size=self.batch_size, uniform_batches=True,
                                             shuffle=True)
            self.val_ds = BilingualDataset(val_ds_raw, self.src_lang, self.tgt_lang, rd.src_tokenizer,
                                           rd.tgt_tokenizer, batch_size=self.batch_size, uniform_batches=True,
                                           shuffle=False)
            del train_ds_raw, val_ds_raw

            self.transformer = Transformer(rd.src_tokenizer.get_vocab_size(), rd.tgt_tokenizer.get_vocab_size())
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing,
                                                 ignore_index=self.train_ds.pad_token)

    def configure_optimizers(self) -> dict:
        # Effective LR and batch size are different in DDP
        effective_lr = self.learning_rate * utils.get_device()[1]
        optimizer = optim.Adam(self.parameters(), lr=effective_lr, eps=1e-9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss"
            }
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=self.train_ds.collate_fn,
                          sampler=self.train_ds.sampler, num_workers=os.cpu_count(), pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, collate_fn=self.val_ds.collate_fn, shuffle=False,
                          num_workers=os.cpu_count(), pin_memory=True)

    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.enable_gc == 'batch':
            garbage_collection_cuda()

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if self.enable_gc == 'batch':
            garbage_collection_cuda()

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if self.enable_gc == 'batch':
            garbage_collection_cuda()

    def get_current_lrs(self):
        try:
            return ', '.join(str(param_group['lr']) for param_group in self.optimizers().optimizer.param_groups)
        except AttributeError:
            return 'Not Available'

    def on_train_epoch_end(self):
        if self.enable_gc == 'epoch':
            garbage_collection_cuda()
        print(f"Epoch: {self.current_epoch}, Global Steps: {self.global_step}, "
              f"Train Loss: {self.my_train_loss.compute()}, LR: {self.get_current_lrs()}")
        self.my_train_loss.reset()

    def on_validation_epoch_end(self):
        if self.enable_gc == 'epoch':
            garbage_collection_cuda()
        print(f"Epoch: {self.current_epoch}, Global Steps: {self.global_step}, "
              f"Val Loss: {self.my_val_loss.compute()}, LR: {self.get_current_lrs()}")
        self.my_val_loss.reset()

    def on_predict_epoch_end(self):
        if self.enable_gc == 'epoch':
            garbage_collection_cuda()
