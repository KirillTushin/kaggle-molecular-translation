import torch
from torch.nn import functional as F

from catalyst import dl
from catalyst.utils import any2device


class CustomRunner(dl.Runner):
    
    def predict_batch(self, batch):
        batch      = any2device(batch, self.device)
        images     = batch['images']
        batch_size = images.shape[0]

        image_embeddings = self.model.image_embeddings(images)
        image_embeddings = self.model.transformer.encoder(image_embeddings)
        
        
        tokens = [[self.model.start_idx]]*batch_size
        tgt = torch.LongTensor(tokens).to(self.device)
        for _ in range(self.model.max_len):
            
            tgt_mask = self.model.make_tgt_mask(tgt)
            tgt_key_padding_mask = self.model.make_pad_mask(tgt)
            
            token_embeddings = self.model.token_embeddings(tgt)

            output = self.model.transformer.decoder(token_embeddings, image_embeddings, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            pred   = self.model.linear(output).transpose(1, 0)
            
            last = pred[:,-1:].argmax(dim=-1)
            
            tgt = torch.cat([tgt, last], dim=1)

        return [list(x) for x in tgt.detach().cpu().numpy()]

    
    def _handle_batch(self, batch):
        batch_metrics = {}
        
        batch = any2device(batch, self.device)

        tokens = batch['texts']
        
        batch['tokens'] = tokens[:, :-1]
        
        y     = tokens[:,1:]
        y_hat = self.model(batch).transpose(2, 1)

        loss = F.cross_entropy(y_hat, y, ignore_index=self.model.pad_idx)
        
        self.batch_metrics.update(
            {"loss": loss}
        )

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

