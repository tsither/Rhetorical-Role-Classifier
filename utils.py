<<<<<<< HEAD
import torch.nn as nn
import os, sys
from typing import Tuple, List, Dict
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights() -> torch.FloatTensor:
    sample_input, sample_target = None, None
    for idx in range(246):
        if sample_input is None:
            sample_input = load_tensor(filepath=f"train_document/doc_{idx}/embedding")
            sample_target = load_tensor(filepath=f"train_document/doc_{idx}/label")
        else:
            sample_input = torch.cat((sample_input,load_tensor(filepath=f"train_document/doc_{idx}/embedding")), dim=0)
            sample_target = torch.cat((sample_target,load_tensor(filepath=f"train_document/doc_{idx}/label")), dim=0)

    sample_target = sample_target.long() 
    y_train = sample_target.numpy()
    
    class_weights = compute_class_weight(class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = y_train)
    class_weights = torch.FloatTensor(class_weights)
    return class_weights
def organize_data(data, batch_size:int = 1):
    """
    - Reads training/test data from Dataset_Reader class
    - Organizes data into document indexes, text, and labels for processing

    Parameters:
    - train/test data (python class)
    - batch_size (int) : set to 1 to process all data

    Returns:
    - doc_idx
    - batched_texts
    - batched_labels 
    """
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



def data_to_embeddings(indexes:List, texts:List, labels:List, encoder:LabelEncoder,max_len_dict:Dict,tokenizer= None, emb_model= None,) -> Tuple[torch.TensorType, torch.TensorType]:
    """
    Generates sentence embeddings. Able to handle multiple documents, but in our training process we generate and save embeddings in documents one at a time

    Parameters:
    - indexes (list) : indexes of each document
    - texts (list) : text data of each document
    - labels (list) : labels for each document
    - encoder (class object) : object to encode labels numerically
    - max_len_dict (dict) : dictionary containing the max sentence lengths for each document
    - tokenizer (class) : tokenizer
    - emb_model (class) : model to generate word (and sentence) embeddings


    Returns:
    x_train (pytorch tensor) : sentence embeddings for data
    y_train (pytorch tensor) : labels for data

    """

    numerical_labels = encoder.transform(labels)
    sent_emb = []
    
    for idx, sentence in enumerate(texts):
        try:
            max_sent_length = max([max_len_dict[i] for i in indexes])           #calculate the maximum sentence length with current document
        except KeyError:
            continue

        inputs = tokenizer(sentence,  return_tensors="pt", truncation= True,            #tokenize input for embedding generation      
                                padding='max_length', max_length = max_sent_length,
                                add_special_tokens= True)
        with torch.no_grad():
            output = emb_model(**inputs)                        #create embedding
        sent_emb.append(output.last_hidden_state[:,0,:])        #get pooled sentence embedding 

    x_train = np.zeros((len(sent_emb), 1, 768), dtype=float)        #instantiate objects to store embeddings
    y_train = torch.from_numpy(numerical_labels)

    for idx, sentence in enumerate(sent_emb):               #populate embedding object
        x_train[idx] = sent_emb[idx]
    x_train = torch.from_numpy(x_train).float()
    print(f"X_train size: {x_train.size()}\tY_train size: {y_train.size()}")
    return x_train, y_train   


def init_weights(m):
    """
    - Initialize weights for a pytorch model 
    Parameters:
    - m (pytorch model)
    Returns:
    None
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def save_model(model, filepath):

    """
    Save PyTorch model to a file.

    Parameters:
    - model (torch.nn.Module): PyTorch model to save.
    - filepath (str): Filepath to save the model parameters.
    Returns:
    None
=======
"""Module containing utils functions and classes"""

import os, sys
from typing import Tuple
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight

def save_model(model, filepath):
    """
    Save PyTorch model parameters to a file.

    Args:
    - model (torch.nn.Module): PyTorch model to save.
    - filepath (str): Filepath to save the model parameters.
>>>>>>> main
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model parameters saved to '{filepath}'")

<<<<<<< HEAD

=======
>>>>>>> main
def load_model(model, filepath):
    """
    Load PyTorch model parameters from a file.

<<<<<<< HEAD
    Parameters:
    - model (torch.nn.Module): pytorch model to load parameters into.
    - filepath (str): Filepath to the saved model parameters.
    Returns:
    - m (pytorch model) : loaded pytorch model
    """
    m = model.load_state_dict(torch.load(filepath))
    print(f"Model parameters loaded from '{filepath}'")
    return m
    
    
def save_tensor(tensor, dir, filename):
    """
    Save pytorch tensor to a file.

    Parameters:
    - tensor (torch.Tensor): pytorch tensor to save.
    - dir (str): Directory to save the tensor.
    - filename (str): Filename to save the tensor.
    Returns:
    None
=======
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
>>>>>>> main
    """
    
    if not os.path.exists(os.path.join(dir)):
        os.makedirs(os.path.join(dir))
        
    filepath = os.path.join(dir, filename)
    torch.save(tensor, filepath)
<<<<<<< HEAD
    print(f"Tensor saved to '{filepath}'")

def load_tensor(filepath):
    """
    Load pytorch tensor from a file

    Parameters:
    - filepath (str): filepath to the saved tensor

    Returns:
    - tensor (torch.Tensor): loaded pytorch tensor
=======
    # print(f"Tensor saved to '{filepath}'")

def load_tensor(filepath):
    """
    Load PyTorch tensor from a file.

    Args:
    - filepath (str): Filepath to the saved tensor.

    Returns:
    - tensor (torch.Tensor): Loaded PyTorch tensor.
>>>>>>> main
    """
    tensor = torch.load(filepath)
    # print(f"Tensor loaded from '{filepath}'")
    return tensor

<<<<<<< HEAD

def read_json(FILEPATH, type='r', reading_max_length=False):
    """
    Read json file from filepath, return data as python dictionary

    Parameters:
    - filepath (str) : filepath to json file
    Returns:
    data (dict) : data from json file
    """
    with open(FILEPATH, type) as file:
        data = json.load(file)

    #Covert the keys to integers for efficient document processing if being used to create max length dictionary
    if reading_max_length:
        if isinstance(data, dict):
            new_data = {int(key): value for key, value in data.items()}
            return new_data

    return data

def label_encode(target_variables : list) -> LabelEncoder:
    """
    Encode target variables
=======
def label_encode(target_variables : list) -> LabelEncoder:
    """
    Encode target variables.
>>>>>>> main
    
    Args:
    - target_variables (list or array-like): List of target variable strings
    
    Returns:
    - le (object): class object used to tranform and inverse transform
    """
    le = LabelEncoder()
    le = le.fit(target_variables)
    
    return le
<<<<<<< HEAD


def document_max_length(documents, tokenizer):
    """
    Generate the maximum length of each sentence in each document (necessary to make sure there is a fixed sentence-length 
    for each document before we pass the sentence embeddings through the model)

    Parameters:
    - batch of total training or test documents (python class)

    Returns: 
    - max_length_dict (dict): {document index: length of longest sentence}

    """
    max_length_dict = {}
    for index, sentences in documents.dict.items():         #in data class, grab items from dictionary containing data for all the documents
        sizes = []

        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
            sizes.append(inputs['input_ids'].size(1))

        max_length_dict[index] = max(sizes)

    return max_length_dict


def write_dictionary_to_json(dictionary, file_path):
    """

    create a json file from a dictionary

    Parameters:
    - dictionary : dictionary to be written to the json file
    - file_path: path to the json file

    Returns:
    - None
    """
    with open(file_path, 'w') as json_file:
        json.dump(dictionary, json_file, indent=2)

    print(f"Successfully wrote dictionary to JSON file: {file_path}")


=======

def max_length(documents, tokenizer):
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
        text = [entry[-1] for entry in self.texts if entry[0] == idx]
        label = [entry[-2] for entry in self.texts if entry[0] == idx]

        return {
            'text': text,
            'label': label,
        }
        
def get_class_weights(y_train) -> torch.FloatTensor:
    class_weights = compute_class_weight(class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = y_train)
    class_weights = torch.FloatTensor(class_weights)
    return class_weights
>>>>>>> main
