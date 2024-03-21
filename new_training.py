import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Dataset_Reader import Dataset_Reader
from sklearn.preprocessing import LabelEncoder
import json


TRAIN_DATA_PATH = 'data/train.json'
TEST_DATA_PATH = 'data/dev.json'

data_train = Dataset_Reader(TRAIN_DATA_PATH)
data_test = Dataset_Reader(TEST_DATA_PATH)

print(data_train.texts)