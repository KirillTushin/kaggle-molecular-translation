import math

import torch
from torch import nn
import segmentation_models_pytorch as smp
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, AutoModelForSeq2SeqLM

class ImagePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)


class ImageEmbeddings(nn.Module):
    def __init__(self, image_size, backbone='resnet18', level=-2, emb_dim=128):
        super().__init__()
        self.backbone = smp.encoders.get_encoder(backbone, in_channels=1)
        self.level    = level
        
        random_input = torch.rand(1, 1, image_size[0], image_size[1])
        shape = self.backbone(random_input)[level].shape[1:]
        
        self.seq_len = shape[1]*shape[2]
        self.after_backbone = shape[0]
        
        self.linear      = nn.Linear(self.after_backbone, emb_dim)
        self.pos_encoder = ImagePositionalEncoding(d_model=emb_dim,  max_len=self.seq_len)
    
    def forward(self, x):
        x = self.backbone(x)[self.level]
        x = x.reshape(-1, self.seq_len, self.after_backbone)
        x = self.linear(x) ## BS x NumTokens x EmbDim
        return self.pos_encoder(x)


class Model(nn.Module):
    def __init__(
        self,
        image_size=(256, 256),
        backbone='resnet18',
        level=-1,
        hidden_size=128,
        num_hidden_layers=6,
        num_attention_heads=8,
        max_len=150,
        vocab_size=500,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        
        super().__init__()
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.max_len   = max_len
        
        self.image_embeddings = ImageEmbeddings(image_size, backbone, level, hidden_size)

        config_encoder = BertConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            intermediate_size=hidden_size*4,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=1,
        )
        
        config_decoder = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size*4,
            max_position_embeddings=300,
        )

        config_decoder.is_decoder = True
        config_decoder.add_cross_attention = True

        config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
        
        self.transformer = EncoderDecoderModel(config=config)    
        
    def forward(self, batch):
        images = batch['images']
        tokens = batch['tokens']
        attention_mask = batch['attention_mask']
        
        inputs_embeds = self.image_embeddings(images)
        
        return self.transformer(
            inputs_embeds=inputs_embeds,
            decoder_input_ids=tokens,
            decoder_attention_mask=attention_mask,
            labels=tokens,
        )
