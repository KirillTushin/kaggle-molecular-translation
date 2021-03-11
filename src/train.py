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

from module.model import Model
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
    model = Model(
        image_size          = (config.images.size, config.images.size),
        backbone            = config.model.backbone,
        level               = config.model.level,
        hidden_size         = config.model.hidden_size,
        num_hidden_layers   = config.model.num_hidden_layers,
        num_attention_heads = config.model.num_attention_heads,
        max_len             = config.model.max_len,
        vocab_size          = tokenizer.get_vocab_size(),
        bos_token_id        = tokenizer.token_to_id('[SOS]'),
        eos_token_id        = tokenizer.token_to_id('[EOS]'),
        pad_token_id        = tokenizer.token_to_id('[PAD]'),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.T_max)


    log.info('Train Model')
    runner = CustomRunner(model=model, device=config.device)

    runner.train(
        model      = model,
        optimizer  = optimizer,
        scheduler  = scheduler,
        loaders    = loaders,
        logdir     = config.training.logdir,
        num_epochs = config.training.num_epochs,
        resume     = config.training.resume if config.training.resume else None,
        fp16       = config.training.fp16 if config.training.fp16 else None,
        verbose    = True,
    )


if __name__ == "__main__":
    main()
