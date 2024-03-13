from torch.utils.data import Dataset
from preprocessing import remove_special_characters, remove_stopwords
import json
###########################################################################

# file defining the data reader

###########################################################################



class Dataset_Reader(Dataset):
    def __init__(self, data_path, remove_sc=False):
        self.data_path = data_path
        self.dict = {}
        self.texts = []
        self.labels = []
        self.doc_count = 0

        with open(data_path, 'r') as file:
            data = json.load(file)

        for idx, document in enumerate(data):
            self.doc_count+=1
            # current_id = document['id']
            # current_meta = document['meta']['group']
            for annotation in document['annotations']:
                for sentence in annotation['result']:
                    text = sentence['value']['text'].lower().replace('\n', '')

                    if remove_sc== True: #remove special characters
                        cleaned_text = remove_special_characters(text) 
                        label = sentence['value']['labels'][0]

                        self.texts.append([idx, label, cleaned_text])
                        self.labels.append(label)

                    else: #include special characters
                        label = sentence['value']['labels'][0]

                        self.texts.append([idx, label, text])
                        self.labels.append(label)


        for sentence in self.texts:
            key = sentence[0]
            value = sentence[2]
            if key in self.dict:
                self.dict[key].append(value)
            
            else:
                self.dict[key] = [value]
             
        



    def __len__(self):
            return self.doc_count

    def __getitem__(self, idx):
        text = [entry[-1] for entry in self.texts if entry[0] == idx]
        label = [entry[-2] for entry in self.texts if entry[0] == idx]

        return {
            'text': text,
            'label': label,
        }