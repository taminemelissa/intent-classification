import uuid
from tqdm import tqdm
from typing import List, Tuple
from src.data.data_format import Utterance


def convert_transformers_dataset_to_utterances(dataset) -> List[Utterance]:
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
            utterance = Utterance(text=text, identifier=identifier, da=da, label=label, meta=meta)
        if 'Sentiment' in dataset[i].keys():
            sentiment = dataset[i]['Sentiment']
            meta.pop('Sentiment')
            utterance = Utterance(text=text, identifier=identifier, sentiment=sentiment, label=label, meta=meta)
        utterances.append(utterance)
    print(f"################ {len(dataset['Utterance'])} formated #################")
    return utterances


def generate_batches(input: List[object], batch_size) -> List[object]:
    for i in tqdm(range(0, len(input), batch_size)):
        yield input[i:i+batch_size]

    
def process_utterance(u: Utterance) -> Tuple[str, int]:
    utterance = u.text
    label = int(u.label)
    return (utterance, label)
