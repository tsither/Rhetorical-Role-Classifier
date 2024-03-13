from transformers import BertTokenizer, BertModel
from Dataset_Reader import Dataset_Reader
from utils import read_json, get_model_data_batched, save_tensor, label_encode, get_batched_data
from utils import document_max_length, write_dictionary_to_json

from main import TRAIN_DATA_PATH, TEST_DATA_PATH


##################################################################################

# File relating to the process of reading the data and generating word embeddings

##################################################################################

 #initialize relevant libraries to compute word embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')                          
emb_model = BertModel.from_pretrained('bert-base-uncased')

#retrieve data, choose whether to include special characters in data
train_data = Dataset_Reader(TRAIN_DATA_PATH, include_special_characters=True)
test_data = Dataset_Reader(TEST_DATA_PATH, include_special_characters=True)

print(f"Number of sentences in training data: {len(train_data.texts)}")
print(f"Number of sentences in test data: {len(test_data.texts)}")

# Manually defining labels for convenience
list_of_targets = ['ISSUE', 'FAC', 'NONE', 'ARG_PETITIONER', 'PRE_NOT_RELIED', 'STA', 'RPC', 'ARG_RESPONDENT', 'PREAMBLE', 'ANALYSIS', 'RLC', 'PRE_RELIED', 'RATIO']

# Numerically encode labels
label_encoder = label_encode(list_of_targets)


#Compute the maximum sentence length for each document in the training and test data (to ensure all embeddings will be the same size within a document)
max_length_dict_TRAIN = document_max_length(train_data, tokenizer=tokenizer)
max_length_dict_TEST = document_max_length(test_data, tokenizer=tokenizer)

# To same time during training process, write these documents to json file
write_dictionary_to_json(max_length_dict_TRAIN, 'max_length_dicts/max_length_train.json')
write_dictionary_to_json(max_length_dict_TEST, 'max_length_dicts/max_length_test.json')


#retrieve max_length dictionaries to compute word embeddings
max_length_dict_TRAIN = read_json('max_length_dicts/max_length_train.json', reading_max_length=True)
max_length_dict_TEST = read_json('max_length_dicts/max_length_test.json', reading_max_length=True)


#organize and process data
doc_idxs, batched_texts, batched_labels = get_batched_data(train_data, batch_size= 1) 


for idx in range(len(doc_idxs)):
    TRAIN_emb, TRAIN_labels = get_model_data_batched(doc_idxs[idx], batched_texts[idx], batched_labels[idx],label_encoder,max_length_dict_TRAIN, tokenizer=tokenizer, emb_model=emb_model)
    save_tensor(TRAIN_emb, 'test_document/doc_'+str(idx),"embedding")
    save_tensor(TRAIN_labels, 'test_document/doc_'+str(idx),"label")


# #####CURRENTLY ADAPTING FOR DIFFERENT PREPROCESSING FEATURES
# for idx in range(len(doc_idxs)):
#     TRAIN_emb, TRAIN_labels = get_model_data_batched(doc_idxs[idx], batched_texts[idx], batched_labels[idx],label_encoder,max_length_dict_TRAIN, tokenizer=tokenizer, emb_model=emb_model)
#     save_tensor(TRAIN_emb, 'tensors_no_special_characters/doc_'+str(idx),"embedding")
#     save_tensor(TRAIN_labels, 'tensors_no_special_characters/doc_'+str(idx),"label")

