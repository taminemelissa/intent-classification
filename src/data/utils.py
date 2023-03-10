import uuid
from tqdm import tqdm
from typing import List
from src.data.data_format import *

def convert_transformers_dataset_to_passages(dataset) -> List[Utterance]:
    """
    This function converts a Transformer Dataset into a list of Utterance objects
    """
    print('######## Converting dataset into a list of utterances ########')
    utterances = []
    for i in tqdm(range(len(dataset))):
        text = dataset[i]['Utterance']
        identifier = uuid.uuid1().hex
        label = dataset[i]['Label']
        meta = dataset[i]
        meta.pop('Utterance')
        meta.pop('Label')
        if 'Dialogue_Act' in dataset[i].keys():
            da = dataset[i]['Dialogue_Act']
            meta.pop('Dialogue_Act')
            utterance = Utterance(text = text, identifier = identifier, da = da, label = label, meta = meta)
        if 'Sentiment' in dataset[i].keys():
            sentiment = dataset[i]['Sentiment']
            meta.pop('Sentiment')
            utterance = Utterance(text = text, identifier=identifier, sentiment = sentiment, label = label, meta = meta)
        utterances.append(utterance)
    print(f"################ {len(dataset['Utterance'])} formated #################")
    return utterances

def generate_batches(input: List[object], batch_size) -> List[object]:
    for i in tqdm(range(0, len(input), batch_size)):
        yield input[i:i+batch_size]

    
def split_into_train_test_val_sets(utterances: List[Utterance], train_ratio: float = 0.7, test_ratio: float=0.1) -> dict:
    total = len(utterances)
    train_count = int(train_ratio * total)
    test_count = int(test_ratio * total)
    if total - train_count - test_count > 0:
        random.shuffle(utterances)
        train_utterances = utterances[0:train_count]
        test_utterances = utterances[train_count:train_count + test_count]
        val_utterances = utterances[train_count + test_count:]
        return {'train': train_utterances, 'test': test_utterances, 'val': val_utterances}
    else:
        raise Exception('Set correct ratios')