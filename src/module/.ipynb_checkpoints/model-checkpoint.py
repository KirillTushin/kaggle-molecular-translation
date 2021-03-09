import math

import torch
from torch import nn

import segmentation_models_pytorch as smp

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ImageEmbeddings(nn.Module):
    def __init__(self, img_size, backbone='resnet18', level=-2, emb_dim=128):
        super().__init__()
        self.backbone = smp.encoders.get_encoder(backbone, in_channels=1)
        self.level    = level
        
        random_input = torch.rand(1, 1, img_size[0], img_size[1])
        shape = self.backbone(random_input)[level].shape[1:]
        
        self.seq_len = shape[1]*shape[2]
        self.after_backbone = shape[0]
        
        self.linear      = nn.Linear(self.after_backbone, emb_dim)
        self.pos_encoder = PositionalEncoding(d_model=emb_dim,  max_len=self.seq_len)
    
    def forward(self, x):
        x = self.backbone(x)[self.level]
        x = x.reshape(-1, self.seq_len, self.after_backbone)
        x = self.linear(x)     ## BS x NumTokens x EmbDim
        x = x.transpose(1, 0)  ## NumTokens x BS x EmbDim
        return self.pos_encoder(x)


class TokenEmbeddins(nn.Module):
    def __init__(self, num_tokens, emb_dim=128, max_len=150):
        super().__init__()
        self.embeddings  = nn.Embedding(num_tokens, emb_dim)
        self.pos_encoder = PositionalEncoding(d_model=emb_dim,  max_len=max_len)

    def forward(self, x):
        x = self.embeddings(x) ## BS x NumTokens x EmbDim
        x = x.transpose(1, 0)  ## NumTokens x BS x EmbDim
        return self.pos_encoder(x)


class MyModel(nn.Module):
    def __init__(
        self,
        tokenizer,
        img_size   = (256, 256),
        backbone   = 'resnet18',
        level      = -1,
        emb_dim    = 64,
        max_len    = 150,
        device     = 'cuda:2',
    ):
        
        super().__init__()
        num_tokens     = tokenizer.get_vocab_size()
        self.tokenizer = tokenizer
        self.start_idx = tokenizer.token_to_id('[SOS]')
        self.pad_idx   = tokenizer.token_to_id('[PAD]')
        self.max_len   = max_len
        self.device    = device
        
        self.image_embeddings = ImageEmbeddings(img_size, backbone, level, emb_dim)
        self.token_embeddings = TokenEmbeddins(num_tokens, emb_dim, max_len)
        self.transformer    = nn.Transformer(
            d_model         = emb_dim,
            dim_feedforward = emb_dim*4,
        )
        self.linear = nn.Linear(emb_dim, num_tokens)
    
        
    def forward(self, batch):
        images = batch['images']
        tokens = batch['tokens']
        
        tgt_mask = self.make_tgt_mask(tokens)
        tgt_key_padding_mask = self.make_pad_mask(tokens)
        
        image_embeddings = self.image_embeddings(images)
        token_embeddings = self.token_embeddings(tokens)

        output = self.transformer(image_embeddings, token_embeddings, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return self.linear(output).transpose(1, 0)
    
    
    def make_tgt_mask(self, trg):
        return self.transformer.generate_square_subsequent_mask(trg.shape[1]).to(self.device)
    
    def make_pad_mask(self, trg):
        return trg == self.pad_idx
    

