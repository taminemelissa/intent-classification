from src.model.base_models import BaseModel
from typing import Union
from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from src.model.utils import get_device
from config import config


class SequenceClassifierModel(pl.LightningModule, BaseModel):
    def __init__(self, encoder_path: str = None, tokenizer_path: str = None,  
                 config: Union[dict, str] = config, device: str = None):
        super().__init__()
        self.config = config
        self.class_num = self.config.CLASS_NUMBER
        self.encoder_path = self.config.ENCODER_PATH if not encoder_path else encoder_path
        self.classifier = nn.Linear(768, self.class_num)
        self.tokenizer_path = self.config.TOKENIZER_PATH if not tokenizer_path else tokenizer_path
        self.save_hyperparameters
        print(self.hparams)
        self.encoder = AutoModel.from_pretrained(self.encoder_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self._device = get_device() if not device else device
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, utterance_ids, utterance_mask, labels=None):
        utterance_embeddings = self.encoder(input_ids=utterance_ids, attention_mask=utterance_mask)
        
        #Linear classifier
        logits = self.classifier(utterance_embeddings.last_hidden_state[:,0]) #CLS embedding

        return logits, self.loss_fct(logits, labels.flatten())
    
    def training_step(self, batch, batch_idx):
        utterance_ids = batch['utterance_ids']
        utterance_mask = batch['utterance_mask']
        labels = batch['labels']
        _, loss = self(utterance_ids, utterance_mask, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        utterance_ids = batch['utterance_ids']
        utterance_mask = batch['utterance_mask']
        labels = batch['labels']
        _, loss = self(utterance_ids, utterance_mask, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        utterance_ids = batch['utterance_ids']
        utterance_mask = batch['utterance_mask']
        labels = batch['labels']
        _, loss = self(utterance_ids, utterance_mask, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.LEARNING_RATE)

    def predict_class(self, input: str) -> int:
        #Encode each utterance
        utterance_encoding = self.tokenizer(input, return_tensors='pt', padding='max_length', 
                                        truncation=True, max_length=512, add_special_tokens=True)
        
        utterance_encoding.to(self._device)

        #Retrieve the ids and attention masks
        utterance_ids = utterance_encoding['input_ids']
        utterance_mask = utterance_encoding['attention_mask']

        #Compute the model embeddings
        utterance_embeddings = self.encoder(input_ids=utterance_ids, attention_mask=utterance_mask)

        #Predict the utterance class
        logits = self.classifier(utterance_embeddings.last_hidden_state[:,0])
        predicted_class = np.argmax(logits.cpu().numpy()).flatten().item()
        
        return predicted_class