import torch
from torch.utils.data import Dataset, random_split
from datasets import load_dataset, load_from_disk
from pathlib import Path

# Huggingface datasets and tokenizers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


def get_or_build_tokenizer(file_path, ds, lang):
    tokenizer_path = Path(file_path)
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator((item['translation'][lang] for item in ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


class RawDataset(object):
    def __init__(self, ds_name, src_lang, tgt_lang):
        super(RawDataset, self).__init__()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.lang_pair = f'{src_lang}-{tgt_lang}'
        try:
            ds_raw = load_from_disk(f'../data/{ds_name}/{self.lang_pair}')
        except FileNotFoundError:
            print("Dataset not found, downloading it...")
            ds_raw = load_dataset(ds_name, self.lang_pair, split='train')
            ds_raw.save_to_disk(f'../data/{ds_name}/{self.lang_pair}')
        self.dataset = ds_raw
        self.src_tokenizer = get_or_build_tokenizer(f'tokenizer_{self.src_lang}.json', ds_raw, self.src_lang)
        self.tgt_tokenizer = get_or_build_tokenizer(f'tokenizer_{self.tgt_lang}.json', ds_raw, self.tgt_lang)

    def split(self, train_split=0.9, seed=42):
        # Keep 90% for training, 10% for validation
        if train_split > 1:
            train_split *= 0.01
        train_ds_size = int(train_split * len(self.dataset))
        val_ds_size = len(self.dataset) - train_ds_size

        train_ds_raw, val_ds_raw = random_split(self.dataset, [train_ds_size, val_ds_size],
                                                generator=torch.Generator().manual_seed(seed))
        return train_ds_raw, val_ds_raw


class BilingualDataset(Dataset):
    def __init__(self, ds_raw, src_lang, tgt_lang, src_tokenizer, tgt_tokenizer, max_src_len=150, src_tgt_diff=10):
        super(BilingualDataset, self).__init__()
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_src_len = max_src_len
        self.src_tgt_diff = src_tgt_diff

        self.dataset = list()
        for item in ds_raw:
            src_text = item['translation'][src_lang]
            tgt_text = item['translation'][tgt_lang]
            src_tokens = self.src_tokenizer.encode(src_text).ids
            tgt_tokens = self.tgt_tokenizer.encode(tgt_text).ids
            if len(src_tokens) <= self.max_src_len and len(tgt_tokens) <= len(src_tokens) + self.src_tgt_diff:
                self.dataset.append({'src_text': src_text, 'tgt_text': tgt_text, 'src_tokens': src_tokens,
                                     'tgt_tokens': tgt_tokens})
        del ds_raw

        self.sos_token = self.tgt_tokenizer.token_to_id('[SOS]')
        self.eos_token = self.tgt_tokenizer.token_to_id('[EOS]')
        self.pad_token = self.tgt_tokenizer.token_to_id('[PAD]')
        self.sos_tokens = [self.sos_token]
        self.eos_tokens = [self.eos_token]
        self.pad_tokens = [self.pad_token] * (max_src_len + src_tgt_diff)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        src_text = row['src_text']
        tgt_text = row['tgt_text']
        enc_input_tokens = row['src_tokens']
        dec_input_tokens = row['tgt_tokens']

        # Add <s> and </s> token
        encoder_input = self.sos_tokens + enc_input_tokens + self.eos_tokens
        # Add only <s> token
        decoder_input = self.sos_tokens + dec_input_tokens
        # Add only </s> token
        label = dec_input_tokens + self.eos_tokens

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

    @staticmethod
    def causal_mask(seq_len):
        # (seq_len, seq_len)
        return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.int))

    def collate_fn(self, batch):
        max_src_len = max(len(item['encoder_input']) for item in batch)
        max_tgt_len = max(len(item['decoder_input']) for item in batch)
        # max_seq_len = max(max_src_len, max_tgt_len)

        encoder_input = list()
        decoder_input = list()
        label = list()
        src_text = list()
        tgt_text = list()

        for item in batch:
            src_pad_len = max_src_len - len(item['encoder_input'])
            encoder_input.append(torch.tensor(item['encoder_input'] + self.pad_tokens[:src_pad_len], dtype=torch.int64))
            tgt_pad_len = max_tgt_len - len(item['decoder_input'])
            decoder_input.append(torch.tensor(item['decoder_input'] + self.pad_tokens[:tgt_pad_len], dtype=torch.int64))
            label.append(torch.tensor(item['label'] + self.pad_tokens[:tgt_pad_len], dtype=torch.int64))
            src_text.append(item['src_text'])
            tgt_text.append(item['tgt_text'])

        encoder_input = torch.stack(encoder_input, dim=0)
        decoder_input = torch.stack(decoder_input, dim=0)
        label = torch.stack(label, dim=0)

        return {
            "encoder_input": encoder_input,  # (B, e_seq_len)
            "decoder_input": decoder_input,  # (B, d_seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(1).unsqueeze(1).int(),  # (B, 1, 1, e_seq_len)
            "decoder_mask": ((decoder_input != self.pad_token).unsqueeze(1).int()
                             & self.causal_mask(decoder_input.shape[1]).unsqueeze(0)).unsqueeze(1),  # (B, 1, d_seq_len, d_seq_len)
            "label": label,  # (B, d_seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
