import os
import pickle
import logging

import hydra
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from catalyst.utils.misc import set_global_seed

from module.model import MyModel
from module.runner import CustomRunner
from module.dataset import ChemicalDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
log = logging.getLogger(__name__)

@hydra.main(config_name='../configs/train.yaml')
def main(config):
    set_global_seed(config.seed)
    os.chdir(hydra.utils.get_original_cwd())


    log.info('Read Data')
    train = pd.read_csv(f'{config.dataframes.path}/train.csv')
    valid = pd.read_csv(f'{config.dataframes.path}/valid.csv')


    log.info('Read Tokenizer')
    with open(f'{config.tokenizer.path}/{config.tokenizer.file}', 'rb') as f:
        tokenizer = pickle.load(f)


    log.info('Tokenize train and valid')
    train['InChI_tokenized'] = tokenizer.encode_batch(train['InChI'])
    valid['InChI_tokenized'] = tokenizer.encode_batch(valid['InChI'])
    max_len = len(train['InChI_tokenized'].iloc[0])

    log.info('Create Datasets and loaders')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    train_dataset  = ChemicalDataset(
        dataframe  = train,
        image_path = f'resized_image_path',
        transform  = transform,
        target     = True,
    )

    valid_dataset  = ChemicalDataset(
        dataframe  = valid,
        image_path = f'resized_image_path',
        transform  = transform,
        target     = True,
    )

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size  = config.training.batch_size,
            num_workers = config.dataframes.num_workers,
            shuffle     = True,
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_size  = config.training.batch_size,
            num_workers = config.dataframes.num_workers,
            shuffle     = False,
        ),
    }


    log.info('Create Model')
    model = MyModel(
        tokenizer = tokenizer,
        img_size  = (config.images.size, config.images.size),
        backbone  = config.model.backbone,
        level     = config.model.level,
        emb_dim   = config.model.emb_dim,
        device    = config.device,
        max_len   = max_len,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.T_max)


    log.info('Train Model')
    runner = CustomRunner(device=config.device)
    runner.train(
        model      = model,
        optimizer  = optimizer,
        scheduler  = scheduler,
        loaders    = loaders,
        logdir     = config.training.logdir,
        num_epochs = config.training.num_epochs,
        resume     = config.training.resume,
        verbose    = True,
    )


if __name__ == "__main__":
    main()
