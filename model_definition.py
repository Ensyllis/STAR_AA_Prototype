import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, get_linear_schedule_with_warmup

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        contrast_feature = features.view(batch_size, -1, features.shape[-1])
        anchor = contrast_feature[:, 0]
        contrast = contrast_feature[:, 1:]
        
        anchor = nn.functional.normalize(anchor, dim=1)
        contrast = nn.functional.normalize(contrast, dim=-1)
        
        # Compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor.unsqueeze(1), contrast.transpose(-2, -1)).squeeze(1),
                                        self.temperature)
        
        # Compute log-prob
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive
        mask = mask[:, 1:]
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        loss = -mean_log_prob_pos.mean()
        return loss

class AuthorshipModel(pl.LightningModule):
    def __init__(self, model_name, num_classes, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        self.criterion = SupConLoss()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # Use CLS token

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].view(-1, batch['input_ids'].size(-1))
        attention_mask = batch['attention_mask'].view(-1, batch['attention_mask'].size(-1))
        labels = batch['labels']
        
        features = self(input_ids, attention_mask)
        features = features.view(labels.size(0), -1, features.size(-1))
        
        loss = self.criterion(features, labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]