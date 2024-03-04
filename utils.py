"""Module containing utils functions and classes"""

import os, sys
from typing import Tuple
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel

def save_model(model, filepath):
    """
    Save PyTorch model parameters to a file.

    Args:
    - model (torch.nn.Module): PyTorch model to save.
    - filepath (str): Filepath to save the model parameters.
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model parameters saved to '{filepath}'")

def load_model(model, filepath):
    """
    Load PyTorch model parameters from a file.

    Args:
    - model (torch.nn.Module): PyTorch model to load parameters into.
    - filepath (str): Filepath to the saved model parameters.
    """
    model.load_state_dict(torch.load(filepath))
    print(f"Model parameters loaded from '{filepath}'")
    
def save_tensor(tensor, dir, filename):
    """
    Save PyTorch tensor to a file.

    Args:
    - tensor (torch.Tensor): PyTorch tensor to save.
    - dir (str): Directory to save the tensor.
    - filename (str): Filename to save the tensor.
    """
    
    if not os.path.exists(os.path.join(dir)):
        os.makedirs(os.path.join(dir))
        
    filepath = os.path.join(dir, filename)
    torch.save(tensor, filepath)
    # print(f"Tensor saved to '{filepath}'")

def load_tensor(filepath):
    """
    Load PyTorch tensor from a file.

    Args:
    - filepath (str): Filepath to the saved tensor.

    Returns:
    - tensor (torch.Tensor): Loaded PyTorch tensor.
    """
    tensor = torch.load(filepath)
    # print(f"Tensor loaded from '{filepath}'")
    return tensor

def label_encode(target_variables : list) -> LabelEncoder:
    """
    Encode target variables.
    
    Args:
    - target_variables (list or array-like): List of target variable strings.
    
    Returns:
    - lb (object): class object used to tranform and inverse transform.
    """
    le = LabelEncoder()
    le = le.fit(target_variables)
    return le

def get_model_data(data:torch.utils.data.Dataset, encoder: LabelEncoder,
                   tokenizer= BertTokenizer.from_pretrained('bert-base-uncased'),
                   model= BertModel.from_pretrained('bert-base-uncased'),
                   num_of_docs:int = None,
                   ) -> Tuple[torch.TensorType, torch.TensorType]:
    numerical_labels = encoder.transform(data.labels)
    sent_emb = []
    max_sent_length = 128
    if num_of_docs is None:
        for idx, sentence in enumerate(data.texts):
            inputs = tokenizer(sentence[2].lower(),  return_tensors="pt", truncation= True,
                                padding='max_length', max_length = max_sent_length,
                                add_special_tokens= True)
            with torch.no_grad():
                output = model(**inputs)
            sent_emb.append(output.last_hidden_state[:,0,:])
    else:
        for idx, sentence in enumerate(data.texts):
            if sentence[0] < num_of_docs:
                inputs = tokenizer(sentence[2].lower(),  return_tensors="pt", truncation= True,
                                    padding='max_length', max_length = max_sent_length,
                                    add_special_tokens= True)
                with torch.no_grad():
                    output = model(**inputs)
                sent_emb.append(output.last_hidden_state[:,0,:]) 
        numerical_labels = numerical_labels[:len(sent_emb)]
    x_train = np.zeros((len(sent_emb), 1, 768), dtype=float)
    y_train = torch.from_numpy(numerical_labels)
    for idx, sentence in enumerate(sent_emb):
        x_train[idx] = sent_emb[idx]
    x_train = torch.from_numpy(x_train).float()
    print(f"X_train size: {x_train.size()}\nY_train size: {y_train.size()}")
    return x_train, y_train

class Dataset_Reader(Dataset):
    def __init__(self, data):
        self.data = data

        self.texts = []
        self.labels = []
        for idx, document in enumerate(data):
            current_id = document['id']
            # current_meta = document['meta']['group']
            for annotation in document['annotations']:
                for sentence in annotation['result']:
                    text = sentence['value']['text'].lower().replace('\n', '')
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