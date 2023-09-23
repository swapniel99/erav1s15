import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from torchmetrics import MeanMetric
from datasets import load_dataset, load_from_disk

import utils
from .vanilla import Transformer
from dataset import BilingualDataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Model(LightningModule):
    def __init__(self, max_seq_len: int = 350, src_lang: str = 'en', tgt_lang: str = 'it', label_smoothing: float = 0.1,
                 batch_size: int = 32, learning_rate: float = 1e-4, enable_gc='batch'):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.transformer = None
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.max_seq_len = max_seq_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_tokenizer = None
        self.tgt_tokenizer = None
        self.train_ds = None
        self.val_ds = None
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.enable_gc = enable_gc
        self.my_train_loss = MeanMetric()
        self.my_val_loss = MeanMetric()

    def forward(self, encoder_input, encoder_mask, decoder_input, decoder_mask):
        return self.transformer.forward(encoder_input, encoder_mask, decoder_input, decoder_mask)

    def common_forward(self, batch):
        encoder_input = batch['encoder_input']  # (B, seq_len)
        decoder_input = batch['decoder_input']  # (B, seq_len)
        encoder_mask = batch['encoder_mask']  # (B, 1, 1, seq_len)
        decoder_mask = batch['decoder_mask']  # (B, 1, seq_len, seq_len)

        output = self.forward(encoder_input, encoder_mask, decoder_input, decoder_mask)  # (B, seq_len, vocab_size)
        del encoder_input, encoder_mask, decoder_input, decoder_mask
        return output

    def common_step(self, batch):
        output = self.common_forward(batch)
        label = batch['label']  # (B, seq_len)
        loss = self.criterion(output.transpose(1, 2), label)
        del output, label
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.my_train_loss.update(loss, batch['label'].shape[0])
        self.log(f"train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.my_val_loss.update(loss, batch['label'].shape[0])
        self.log(f"val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.common_forward(batch)

    def prepare_data(self) -> None:
        try:
            ds_raw = load_from_disk(f'../data/opus_books/{self.src_lang}-{self.tgt_lang}')
        except FileNotFoundError:
            print("Dataset not found, downloading it...")
            ds_raw = load_dataset('opus_books', f"{self.src_lang}-{self.tgt_lang}", split='train')
            ds_raw.save_to_disk(f'../data/opus_books/{self.src_lang}-{self.tgt_lang}')

        # Build tokenizers
        utils.get_or_build_tokenizer(f'tokenizer_{self.src_lang}.json', ds_raw, self.src_lang)
        utils.get_or_build_tokenizer(f'tokenizer_{self.tgt_lang}.json', ds_raw, self.tgt_lang)

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            ds_raw = load_from_disk(f'../data/opus_books/{self.src_lang}-{self.tgt_lang}')
            self.src_tokenizer = utils.get_or_build_tokenizer(f'tokenizer_{self.src_lang}.json', ds_raw, self.src_lang)
            self.tgt_tokenizer = utils.get_or_build_tokenizer(f'tokenizer_{self.tgt_lang}.json', ds_raw, self.tgt_lang)
            self.transformer = Transformer(self.src_tokenizer.get_vocab_size(), self.tgt_tokenizer.get_vocab_size(),
                                           self.max_seq_len)
            self.criterion.ignore_index = self.tgt_tokenizer.token_to_id('[PAD]')

            # Keep 90% for training, 10% for validation
            train_ds_size = int(0.9 * len(ds_raw))
            val_ds_size = len(ds_raw) - train_ds_size

            train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size],
                                                    generator=torch.Generator().manual_seed(42))

            self.train_ds = BilingualDataset(train_ds_raw, self.src_tokenizer, self.tgt_tokenizer, self.src_lang,
                                             self.tgt_lang,  self.max_seq_len)
            self.val_ds = BilingualDataset(val_ds_raw, self.src_tokenizer, self.tgt_tokenizer, self.src_lang,
                                           self.tgt_lang, self.max_seq_len)

    def configure_optimizers(self):
        # Effective LR and batch size are different in DDP
        effective_lr = self.learning_rate * utils.DEVICE_COUNT
        return optim.Adam(self.parameters(), lr=effective_lr, eps=1e-9)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, num_workers=os.cpu_count(),
                          pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, shuffle=False, num_workers=os.cpu_count(),
                          pin_memory=True)

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
