import os
import pickle
import logging

import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from catalyst.utils.misc import set_global_seed

from module.model import Model
from module.runner import CustomRunner
from module.dataset import ChemicalDataset
from module.utils import drop_after_eos, lev_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
log = logging.getLogger(__name__)

@hydra.main(config_name='../configs/predict.yaml')
def main(config):
    os.chdir(hydra.utils.get_original_cwd())


    log.info('Read Data')
    data = pd.read_csv(f'{config.dataframes.path}/{config.dataframes.file}')
    
    log.info('Read Tokenizer')
    with open(f'{config.tokenizer.path}/{config.tokenizer.file}', 'rb') as f:
        tokenizer = pickle.load(f)

    eos_id = tokenizer.token_to_id('[EOS]')

    log.info('Create Datasets and loaders')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    dataset  = ChemicalDataset(
        dataframe  = data,
        image_path = f'resized_image_path',
        transform  = transform,
        target     = False,
    )
    loader = DataLoader(
        dataset,
        batch_size  = config.predict.batch_size,
        num_workers = config.dataframes.num_workers,
        shuffle     = False,
    )


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
    
    log.info('Predict')
    runner = CustomRunner(
        model  = model,
        device = config.device,
    )
    predictions = runner.predict_loader(
        loader=loader,
        resume=config.predict.model_path,
        fp16=config.predict.fp16 if config.predict.fp16 else None,
    )

    results = []
    for pred in tqdm(predictions, total=len(loader)):
        results.extend([drop_after_eos(p, eos_id) for p in pred])

    data['Predicted_InChI'] = ['InChI=1S/' + x.replace(' ', '').replace('##', '') for x in tokenizer.decode_batch(results)]
    data['InChI'] = 'InChI=1S/' +  data['InChI']

    if config.predict.score:
        log.info('Scoring')
        data['score'] = lev_score('InChI=1S/' + data['Predicted_InChI'], data['InChI'])
        log.info(f'Score: {data["score"].mean()}')

    os.makedirs(config.predict.path, exist_ok=True)
    data.to_csv(f'{config.predict.path}/{config.dataframes.file}', index=False)

    if config.predict.submit:
        sub = data[['image_id', 'Predicted_InChI']].rename(columns={'Predicted_InChI':'InChI'})
        sub.to_csv(f'{config.predict.path}/submit_{config.dataframes.file}', index=False)


if __name__ == "__main__":
    main()
