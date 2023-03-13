from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from src.data.data_format import UtteranceCollection
from torch import Tensor
import pytorch_lightning as pl
from typing import Union, List, Optional
from src.data.utils import process_utterance
from config import config


class SequenceClassifierModelDataset(Dataset):
    def __init__(self, data: UtteranceCollection, tokenizer: BertTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        source = process_utterance(self.data.utterances[index])
        utterance, label = source[0], source[1]
        utterance_encoding = self.tokenizer(utterance, return_tensors='pt', padding='max_length', 
                                            truncation=True, max_length=512, add_special_tokens=True)
        labels = [0]*config.CLASS_NUMBER
        labels[label] = 1
        labels = Tensor(labels)
        return dict(utterance_text=utterance,
                    utterance_ids=utterance_encoding['input_ids'].flatten(),
                    utterance_mask=utterance_encoding['attention_mask'].flatten(),
                    labels=labels.flatten())

class SequenceClassifierModelDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_data: UtteranceCollection,
                 val_data: UtteranceCollection,
                 test_data: UtteranceCollection,
                 tokenizer: BertTokenizer,
                 batch_size: int = 12,
                 num_workers: int = 72):
        super().__init__()
        self.batch_size = batch_size
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        self.train_dataset = SequenceClassifierModelDataset(self.train_data, self.tokenizer)
        self.val_dataset = SequenceClassifierModelDataset(self.val_data, self.tokenizer)
        self.test_dataset = SequenceClassifierModelDataset(self.test_data, self.tokenizer)
    
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)