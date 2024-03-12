from torch.utils.data import Dataset
# from preprocessing import remove_special_characters, remove_stopwords
import json

###########################################################################

# file defining the data reader

###########################################################################



class Dataset_Reader(Dataset):
    def __init__(self, data_path):
        # self.data = data
        self.data_path = data_path
        self.texts = []
        self.labels = []

        with open(data_path, 'r') as file:
            data = json.load(file)

        for idx, document in enumerate(data):
            # current_id = document['id']
            # current_meta = document['meta']['group']
            for annotation in document['annotations']:
                for sentence in annotation['result']:
                    text = sentence['value']['text'].lower().replace('\n', '')

                    # cleaned_text = remove_special_characters(text) #remove special characters
                    label = sentence['value']['labels'][0]

                    self.texts.append([idx, label, text])
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = [entry[-1] for entry in self.texts if entry[0] == idx]
        label = [entry[-2] for entry in self.texts if entry[0] == idx]

        return {
            'text': text,
            'label': label,
        }