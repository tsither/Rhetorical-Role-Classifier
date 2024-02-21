import torch
from torch import TensorType
from sklearn.preprocessing import LabelBinarizer
from transformers import BertTokenizer, BertModel

def label_encode(target_variables : list) -> LabelBinarizer:
    """
    Encode target variables using one-hot encoding.
    
    Args:
    - target_variables (list or array-like): List of target variable strings.
    
    Returns:
    - lb (object): class object used to tranform and inverse transform.
    """
    lb = LabelBinarizer()
    lb = lb.fit(target_variables)
    
    return lb

def sent2tensors(sentence: str, MAX_LEN = None) -> dict:
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # print(tokenizer.tokenize(sentence))
    if MAX_LEN is None:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    else:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding='max_length',max_length = MAX_LEN)
    
    return inputs

def sent2embeddings(sentence: str, MAX_LEN = None) -> TensorType:
    
    model = BertModel.from_pretrained('bert-base-uncased')
    
    inputs = sent2tensors(sentence,MAX_LEN)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state
    
    return emb