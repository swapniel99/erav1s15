import os
import torch
import torchmetrics
from pathlib import Path

# Huggingface datasets and tokenizers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import BilingualDataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_device():
    device_count = 1
    if torch.cuda.is_available():
        device = "cuda"
        device_count = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Devices Found: {device_count} x {device}")
    return device, device_count


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(file_path, ds, lang):
    tokenizer_path = Path(file_path)
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def greedy_decode(model, source, source_mask, device='cpu'):
    tokenizer_tgt = model.tgt_tokenizer
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.shape[1] == model.max_seq_len:
            break

        # build mask for target
        decoder_mask = BilingualDataset.causal_mask(decoder_input.shape[1]).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token for input from last token of output
        prob = model.project(out[:, -1, :])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, val_dataloader, print_msg, writer=None, global_step=0, num_examples=2, device='cpu'):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in val_dataloader:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (B, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (B, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.shape[0] == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = model.tgt_tokenizer.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break

    # Evaluate the character error rate
    # Compute the char error rate
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    print('validation cer', cer)

    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    print('validation wer', wer)

    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    print('validation BLEU', bleu)

    if writer:
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()
