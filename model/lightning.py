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
    def __init__(self, src_lang: str = 'en', tgt_lang: str = 'it', param_sharing: str = None, d_model: int = 512,
                 d_ff: int = 2048, heads: int = 8, dropout: float = 0.1, label_smoothing: float = 0.1,
                 batch_size: int = 32, learning_rate: float = 1e-4, enable_gc='batch', num_epochs=20) -> None:
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.transformer = None
        self.criterion = None
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_tokenizer = None
        self.tgt_tokenizer = None
        self.train_ds = None
        self.val_ds = None
        self.param_sharing = param_sharing
        self.d_model = d_model
        self.d_ff = d_ff
        self.heads = heads
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.enable_gc = enable_gc
        self.num_epochs = num_epochs
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

    def prepare_data(self) -> None:
        RawDataset('opus_books', self.src_lang, self.tgt_lang)

    def setup(self, stage: str) -> None:
        if stage == 'fit' and self.transformer is None:
            rd = RawDataset('opus_books', self.src_lang, self.tgt_lang)
            self.src_tokenizer = rd.src_tokenizer
            self.tgt_tokenizer = rd.tgt_tokenizer

            train_ds_raw, val_ds_raw = rd.split(0.9)
            self.train_ds = BilingualDataset(train_ds_raw, self.src_lang, self.tgt_lang, rd.src_tokenizer,
                                             rd.tgt_tokenizer, batch_size=self.batch_size, uniform_batches=True,
                                             shuffle=True, max_src_len=350, src_tgt_diff=350)
            self.val_ds = BilingualDataset(val_ds_raw, self.src_lang, self.tgt_lang, rd.src_tokenizer,
                                           rd.tgt_tokenizer, batch_size=self.batch_size, uniform_batches=True,
                                           shuffle=False, max_src_len=350, src_tgt_diff=350)
            del train_ds_raw, val_ds_raw

            self.transformer = Transformer(rd.src_tokenizer.get_vocab_size(), rd.tgt_tokenizer.get_vocab_size(),
                                           param_sharing=self.param_sharing, d_model=self.d_model, d_ff=self.d_ff,
                                           heads=self.heads, dropout=self.dropout, max_seq_len=350)
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing,
                                                 ignore_index=self.train_ds.pad_token)

    def configure_optimizers(self) -> dict:
        # Effective LR and batch size are different in DDP
        device_count = utils.get_device()[1]
        effective_lr = self.learning_rate * device_count

        optimizer = optim.Adam(self.parameters(), lr=effective_lr/100, eps=1e-9)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=effective_lr,
            steps_per_epoch=(len(self.train_dataloader()) + device_count - 1) // device_count,
            epochs=self.num_epochs,
            pct_start=0.3,  # 0.2
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy='linear'
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=self.train_ds.collate_fn,
                          sampler=self.train_ds.sampler, num_workers=os.cpu_count(), pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, collate_fn=self.val_ds.collate_fn, shuffle=False,
                          num_workers=os.cpu_count(), pin_memory=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.enable_gc == 'batch':
            garbage_collection_cuda()

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
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
