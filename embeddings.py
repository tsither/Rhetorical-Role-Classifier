
from typing import List
import torch

from transformers import BertTokenizer, BertModel

def sent2tensors(sentence: str, MAX_LEN = None) -> dict:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    if MAX_LEN is None:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    else:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding='max_length',max_length = MAX_LEN)
    return inputs

def sent2embeddings(sentence: List, MAX_LEN = None) -> torch.TensorType:
    model = BertModel.from_pretrained('bert-base-uncased')
    
    inputs = sent2tensors(sentence,MAX_LEN)
    emb = model(**inputs).last_hidden_state
    
    return emb