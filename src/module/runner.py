import torch
from torch.nn import functional as F

from transformers.generation_utils import BeamSearchScorer

from catalyst import dl
from catalyst.utils import any2device


class CustomRunner(dl.Runner):
    def predict_batch(self, batch):
        batch      = any2device(batch, self.device)
        images     = batch['images']
        batch_size = images.shape[0]
        tgt        = torch.LongTensor([[self.model.bos_token_id]]*batch_size).to(self.device)

        image_embeddings         = self.model.image_embeddings(images)
        image_embeddings_encoder = self.model.transformer.encoder(inputs_embeds=image_embeddings)
        

        model_kwargs = {'encoder_outputs':image_embeddings_encoder}


        output = self.model.transformer.greedy_search(
            tgt,
            max_length   = self.model.max_len,
            pad_token_id = self.model.pad_token_id,
            eos_token_id = self.model.eos_token_id,
            **model_kwargs,
        )
        return [list(x) for x in output.detach().cpu().numpy()]

    
    def _handle_batch(self, batch):
        batch_metrics = {}
        batch  = any2device(batch, self.device)
        
        labels = batch['tokens']
        logits = self.model(batch)['logits']
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        
        self.batch_metrics.update(
            {"loss": loss}
        )

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

