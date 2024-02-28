from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from transformers import BertTokenizer, BertModel
import json
import torch
import numpy as np


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def read_json(FILEPATH, type='r'):
    with open(FILEPATH, type) as file:
        data = json.load(file)
    return data


def max_length(documents):
    """
    Generate the maximum length of each sentence in each document. This is necessary to make sure there is a fixed sentence-length 
    for each document before we pass the sentence embeddings through the model.

    Returns: {document index: length of longest sentence}

    """
    max_l_dict = {}
    
    for sentence in documents.texts:
            size = []

            inputs = tokenizer(sentence[2], return_tensors="pt", truncation=True, padding=True)

            size.append(inputs['input_ids'].size(1))
            
            max_l_dict[sentence[0]] = max(size)

    return max_l_dict


def label_encoder(labels, label_encoder=LabelEncoder(), one_hot_encoder=OneHotEncoder(sparse_output=False)):
    """
    Generate one-hot encoded labels
    """
     
    numerical_labels = label_encoder.fit_transform(labels)

    numerical_labels = numerical_labels.reshape(-1,1)

    one_hot_labels = one_hot_encoder.fit_transform(numerical_labels)

    return one_hot_labels


def sentence_embeddings(text_data, max_length_dict, tokenizer, model, doc_idx = 0):
    """
    Generate sentence embeddings 
    ------- Need to amend the way in which it handles each document (current doc_idx isnt intuitive) -------- 
    """
    
    embeddings = []
     
    for sentence in text_data:
        if sentence[0] == doc_idx:
            inputs = tokenizer(sentence[2], return_tensors="pt", truncation=True, padding='max_length',max_length = max_length_dict[sentence[0]])

            with torch.no_grad():
                outputs = model(**inputs)
            emb = outputs.last_hidden_state[ :,0, :]
            embeddings.append(emb)

    return embeddings


def emb_label_to_array(embeddings, one_hot_labels):
    """
    Move embeddings and labels into numpy arrays for training
    """
    X = np.zeros((len(embeddings), 1, 768), dtype=float) #768 represents the BERT embedding dimension
    Y = np.zeros((len(embeddings), 13), dtype=float) #13 represents the number of classes

    for idx, _ in enumerate(embeddings):
        X[idx] = embeddings[idx]
        Y[idx] = one_hot_labels[idx]

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()

    return X, Y


def idx_to_one_hot(idx_encoded_array, output_size):
    """
    Convert an index encoded array to a one_hot_array
    """

    arr = np.array(idx_encoded_array)
    encoded_array = np.zeros((arr.size, output_size), dtype=int)
    encoded_array[np.arange(arr.size),arr] = 1

    return encoded_array










