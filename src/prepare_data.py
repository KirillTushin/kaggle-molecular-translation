import os
import pickle
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from tokenizers import Tokenizer
from tokenizers.models import *
from tokenizers.trainers import *
from tokenizers.pre_tokenizers import Punctuation
from tokenizers.processors import TemplateProcessing

import cv2
from PIL import Image

from module.utils import convert_image_id_2_path

tqdm.pandas()


log = logging.getLogger(__name__)

@hydra.main(config_name='../configs/prepare_data.yaml')
def main(config):
    os.chdir(hydra.utils.get_original_cwd())


    log.info('Read Data')
    data = pd.read_csv(f'{config.data_path}/train_labels.csv')
    test = pd.read_csv(f'{config.data_path}/sample_submission.csv')


    log.info('Change InChI Strings')
    data['InChI'] = data['InChI'].apply(lambda x: x.replace('InChI=1S/', ''))


    log.info('Train Test Split')
    train, valid = train_test_split(
        data,
        test_size    = config.test_size,
        random_state = config.random_state,
        shuffle      = True,
    )

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)


    if config.tokenizer.fit:
        log.info('Create tokenizer file to train')
        os.makedirs(config.tokenizer.path, exist_ok=True)
        
        with open(f'{config.tokenizer.path}/{config.tokenizer.raw}', 'w') as f:
            for line in train['InChI'].values:
                f.write(line + '\n')


        log.info('Create Tokenizer and Train')
        TokenModel = eval(f'{config.tokenizer.model}')
        Trainer    = eval(f'{config.tokenizer.model}Trainer')

        tokenizer = Tokenizer(TokenModel())
        tokenizer.pre_tokenizer = Punctuation()

        trainer = Trainer(
            special_tokens = ['[SOS]', '[EOS]', '[PAD]'],
            vocab_size     = config.tokenizer.vocab_size,
        )
        tokenizer.train(
            files   = [f'{config.tokenizer.path}/{config.tokenizer.raw}'],
            trainer = trainer,
        )

        tokenizer.post_processor = TemplateProcessing(
            single         = '[SOS] $A [EOS]',
            special_tokens = [
                ('[SOS]', tokenizer.token_to_id('[SOS]')),
                ('[EOS]', tokenizer.token_to_id('[EOS]')),
            ],
        )
        tokenizer.enable_padding(
            pad_id    = tokenizer.token_to_id('[PAD]'),
            pad_token = '[PAD]',
        )


        log.info('Save Tokenizer')
        with open(f'{config.tokenizer.path}/{config.tokenizer.file}', 'wb') as f:
            pickle.dump(tokenizer, f)


    if config.images.resize:
        log.info('Change image_paths and create resized_image_paths in dataframes')
        for dataset, path in zip([train, valid, test], ['train', 'train', 'test']):
            dataset['image_path'] = dataset['image_id'].progress_apply(
                lambda x: convert_image_id_2_path(x, f'{config.data_path}/{path}')
            )
            dataset['resized_image_path'] = dataset['image_path'].progress_apply(
                lambda x: x.replace(path, f'{path}_resized_{config.images.size}')
            )

        log.info('Save train, valid, test dataframes')
        os.makedirs(config.dataframes.path, exist_ok=True)
        train.to_csv(f'{config.dataframes.path}/train.csv', index=False)
        valid.to_csv(f'{config.dataframes.path}/valid.csv', index=False)
        test.to_csv(f'{config.dataframes.path}/test.csv', index=False)


        log.info('Resize Images')
        img_size = (config.images.size, config.images.size)
        for dataset, path in zip([train, valid, test], ['train', 'train', 'test']):
            for path, new_path in tqdm(zip(dataset['image_path'], dataset['resized_image_path']), total=len(dataset)):
                img = Image.open(path)
                img = img.resize(img_size, resample=Image.BICUBIC)
                img = cv2.dilate(
                    255 - np.array(img),
                    kernel     = (config.images.dilate.kernel, config.images.dilate.kernel),
                    iterations = config.images.dilate.iterations,
                )
                img = Image.fromarray(img)

                new_path = Path(new_path)
                new_path.parent.mkdir(parents=True, exist_ok=True)

                img.save(new_path)


if __name__ == "__main__":
    main()
