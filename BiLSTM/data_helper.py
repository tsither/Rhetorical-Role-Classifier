
from torch.utils.data import Dataset
import json
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class Dataset(Dataset):
    def __init__(self, data):
        self.data = data

        self.texts = []
        self.labels = []
        

        for idx, document in enumerate(data):
            # current_id = document['id']
            # current_meta = document['meta']['group']
            for annotation in document['annotations']:
                for sentence in annotation['result']:
                    text = sentence['value']['text'].lower().replace('\n', '')
                    labels = sentence['value']['labels'][0]

                    self.texts.append([idx, labels, text])
                    self.labels.append(labels)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        


        return {
            'text': text,
            'label': label,
        }



