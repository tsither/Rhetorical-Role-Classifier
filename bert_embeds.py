from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

import json

file_path = 'data/train.json'

with open(file_path, 'r') as file:
    data = json.load(file)
    
    for key, value in data[0].items():
        
        if key == 'annotations':
            for element in value:
                for key2, value2 in element.items():
                    for element2 in value2:
                        for key3, value3 in element2.items():
                            if isinstance(value3, dict):
                                for key4, value4 in value3.items():
                                    if key4 == "text":
                                        print(value4)
                                    
                                    #should it be tokenizer(value4) here?
                                    inputs = tokenizer(key4, return_tensors="pt", truncation=True, padding=True)
                                    outputs = model(**inputs)

                                    bert_embedding = outputs.last_hidden_state[:, 0, :]


                                    bert_embedding_np = bert_embedding.detach().numpy()

                                    print(bert_embedding_np)