from torch.utils.data import Dataset
from collections import defaultdict

class Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.texts = []
        self.labels = []

        for idx, document in enumerate(data):
            for annotation in document['annotations']:
                for sentence in annotation['result']:
                    text = sentence['value']['text'].lower().replace('\n', '')
                    labels = sentence['value']['labels'][0]

                    self.texts.append([idx, labels, text])
                    self.labels.append(labels)

        self.dict = defaultdict(list)
        for item in self.texts:
            key = item[0]
            self.dict[key].extend(item[2:])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        return {
            'text': text,
            'label': label,
        }




