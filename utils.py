import torch
from torch import TensorType
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel

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

def sent2tensors(sentence: str, MAX_LEN = None) -> dict:
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
    # print(tokenizer.tokenize(sentence))
    if MAX_LEN is None:
        inputs = tokenizer(sentence, return_tensors="pt", truncation= True, padding= True, add_special_tokens= True)
    else:
        inputs = tokenizer(sentence, return_tensors="pt",
                           truncation=True, padding='max_length', 
                           max_length = MAX_LEN, add_special_tokens= True)
    
    return inputs

def sent2wordemb(sentence: str, MAX_LEN = None) -> torch.TensorType:
    model = BertModel.from_pretrained('bert-base-uncased')
    for param in model.parameters():
        param.requires_grad = False
    
    inputs = sent2tensors(sentence,MAX_LEN)
    with torch.no_grad():
        emb = model(**inputs)
    
    return emb[0]

def sent2emb(sentence: str) -> torch.TensorType:
    model = BertModel.from_pretrained('bert-base-uncased')
    for param in model.parameters():
        param.requires_grad = False
    
    inputs = sent2tensors(sentence)
    with torch.no_grad():
        emb = model(**inputs).pooler_output
    
    return emb