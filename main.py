from config import get_config
cfg = get_config()
cfg['batch_size'] = 2
cfg['preload'] = None
cfg['num_epochs'] = 10

from train import train_model

train_model(cfg)
