from torch.utils.data import DataLoader
from dataset import RawDataset, BilingualDataset

rd = RawDataset('opus_books', 'en', 'it')
# train, test = rd.split(0.9)

bds = BilingualDataset(rd.dataset, 'en', 'it', rd.src_tokenizer, rd.tgt_tokenizer, 1000, 100)

dl = DataLoader(bds, batch_size=2, shuffle=True, collate_fn=bds.collate_fn)

batch = next(iter(dl))
