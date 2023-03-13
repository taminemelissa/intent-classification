from typing import Dict, List, Any, OrderedDict, Union
import collections
import random
import json
from tqdm import tqdm


def ordered_dict(obj: dict) -> Union[OrderedDict, object]:
    if type(obj) == 'dict':
        return collections.OrderedDict(obj)
    else:
        return obj


class Utterance:
    def __init__(self, 
                 text: str = None,
                 identifier: str = None,
                 da: str = None,
                 sentiment: str = None,
                 label: int = None,
                 meta: Dict[str, Any] = None):
        self.text = text
        self.identifier = identifier
        self.da = da
        self.sentiment = sentiment
        self.label = label
        self.meta = meta

    def to_dict(self) -> Dict:
        res = {}
        if self.text:
            res.update({'text': self.text})         
        if self.identifier:
            res.update({'identifier': self.identifier})
        if self.da:
            res.update({'da': self.da})
        if self.sentiment:
            res.update({'sentiment': self.sentiment})
        if self.label:
            res.update({'label': self.label})
        if self.meta:
            res.update({'meta': self.meta})
        return res
    
    def from_dict(self, d: Dict):
        text = self.text
        identifier = self.identifier
        da = self.da
        sentiment = self.sentiment
        label = self.label
        meta = self.meta
        if 'text' in d.keys():
            text = d['text']
        if 'identifier' in d.keys():
            identifier = d['identifier']
        if 'da' in d.keys():
            da = d['da']
        if 'sentiment' in d.keys():
            sentiment = d['sentiment']
        if 'label' in d.keys():
            label = d['label']
        if 'meta' in d.keys():
            meta = d['meta']

        return Utterance(text=text, identifier=identifier, da=da, sentiment=sentiment, label=label, meta=meta)


class UtteranceCollection:
    def __init__(self, utterances: List[Utterance] = None,
                 meta: Dict[str, Any] = None):
        """
        :param utterances: a list of Utterances objects
        :param meta: the meta information associated with the collection
        """
        self.utterances = utterances
        self.meta = meta

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index):
        return self.utterances[index]

    def random_subset(self, k:int) -> List[Utterance]:
        """
        Return a random subset of length k of the collection of utterances
        """
        indices = random.choices(range(0, len(self.sentences)), k=k)
        res = []
        for i in indices:
            res.append(self.utterances[i])
        return res

    def to_dict(self) -> Dict:
        res = {}
        if self.utterances:
            utterances = []
            for utterance in self.utterances:
                utterances.append(utterance.to_dict())
            res.update({'utterances': utterances})
        if self.meta:
            res.update({'meta': self.meta})
        return res

    def save(self, output_path: str):
        print(f'####### Save utterance collection to {output_path} #########')
        with open(output_path, 'w', encoding='utf8') as out:
            json.dump(self.to_dict(), out, ensure_ascii=False, indent=None)

    def load_from_json(self, input_filename: str):
        with open(input_filename, 'r', encoding='utf8') as f:
            d = json.load(f)
        if 'meta' in d.keys():
            meta = d['meta']
        utterances = d['utterances']
        utrs = []
        print("############# Loading utterances from json file #################")
        for utterance in tqdm(utterances):
            u = Utterance()
            u = u.from_dict(d=utterance)
            utrs.append(u)
        return UtteranceCollection(utterances=utrs, meta=meta)