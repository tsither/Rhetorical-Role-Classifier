import torch.nn as nn
import os, sys
from typing import Tuple, List, Dict
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'..')))




def get_batched_data(data, batch_size:int = 1):
    doc_idx = []
    batched_texts = []
    batched_labels = []
    for start, stop in zip(range(0,len(data)-batch_size,batch_size), range(batch_size,len(data),batch_size)):
        idxs = []
        texts = []
        labels = []
        for idx in range(start,stop):
            idxs.append(idx) 
            [texts.append(text) for text in data[idx]['text']]
            [labels.append(label) for label in data[idx]['label']]
        
        doc_idx.append(idxs)
        batched_texts.append(texts)
        batched_labels.append(labels)
    return doc_idx, batched_texts, batched_labels




def get_model_data_batched(indexes:List, texts:List, labels:List, encoder:LabelEncoder,max_len_dict:Dict,
                           tokenizer= None,
                           emb_model= None,
                           ) -> Tuple[torch.TensorType, torch.TensorType]:
    numerical_labels = encoder.transform(labels)
    sent_emb = []
    for idx, sentence in enumerate(texts):
        try:
            max_sent_length = max([max_len_dict[i] for i in indexes])
        except KeyError:
            continue
        inputs = tokenizer(sentence[2],  return_tensors="pt", truncation= True,
                            padding='max_length', max_length = max_sent_length,
                            add_special_tokens= True)
        with torch.no_grad():
            output = emb_model(**inputs)
        sent_emb.append(output.last_hidden_state[:,0,:])
    x_train = np.zeros((len(sent_emb), 1, 768), dtype=float)
    y_train = torch.from_numpy(numerical_labels)
    for idx, sentence in enumerate(sent_emb):
        x_train[idx] = sent_emb[idx]
    x_train = torch.from_numpy(x_train).float()
    print(f"X_train size: {x_train.size()}\tY_train size: {y_train.size()}")
    return x_train, y_train   


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


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
    print(f"Tensor saved to '{filepath}'")

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


def read_json(FILEPATH, type='r', reading_max_length=False):
    with open(FILEPATH, type) as file:
        data = json.load(file)

    #Covert the keys to integers for efficient document processing
    if reading_max_length:
        if isinstance(data, dict):
            new_data = {int(key): value for key, value in data.items()}
            return new_data

    return data

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


def document_max_length(documents, tokenizer):
    """
    Generate the maximum length of each sentence in each document. This is necessary to make sure there is a fixed sentence-length 
    for each document before we pass the sentence embeddings through the model.

    Returns: {document index: length of longest sentence}

    """
    max_length_dict = {}
    for index, sentences in documents.dict.items():
        sizes = []

        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
            sizes.append(inputs['input_ids'].size(1))

        max_length_dict[index] = max(sizes)

    return max_length_dict


def write_dictionary_to_json(dictionary, file_path):
    """
    Write a dictionary to a JSON file.

    Parameters:
    - dictionary : dictionary to be written to the JSON file.
    - file_path: path to the JSON file.

    Returns:
    - None
    """
    with open(file_path, 'w') as json_file:
        json.dump(dictionary, json_file, indent=2)

    print(f"Successfully wrote dictionary to JSON file: {file_path}")

    pass

