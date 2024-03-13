from BiLSTM import BiLSTM
from data_helper import Dataset
from utils import read_json, label_encoder, max_length, sentence_embeddings, emb_label_to_array, idx_to_one_hot
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score
import numpy as np



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')                          #initialize libraries
model = BertModel.from_pretrained('bert-base-uncased')



parameters = {'input_size': 768, 'hidden_size': 128, 'output_size': 13, 'num_layers':2}
input_size = 768  
hidden_size = 128
output_size = 13  
num_layers = 2
num_epochs = 300
lr = 0.001



train_dataset = Dataset(read_json("./data/train.json"))                 #Load data
test_dataset = Dataset(read_json("./data/dev.json"))


one_hot_train = label_encoder(train_dataset.labels)                           #encode labels
one_hot_test = label_encoder(test_dataset.labels)

print("\n Computing max length dictionary...\n ")
max_length_train = max_length(train_dataset)
max_length_test = max_length(test_dataset)

print("\n Generating sentence embeddings...\n")
train_emb = sentence_embeddings(train_dataset.texts, max_length_train, tokenizer, model=model)
test_emb = sentence_embeddings(test_dataset.texts, max_length_test, tokenizer, model=model)

X_train, Y_train = emb_label_to_array(train_emb, one_hot_train)
X_test, Y_test = emb_label_to_array(test_emb, one_hot_test)




model = BiLSTM(**parameters)                                            # Initialize model              

criterion = nn.CrossEntropyLoss()                                       #Initialize loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

print("\nTraining model...\n")

for epoch in range(num_epochs):
        output_probabilities = model(X_train)

        labels = torch.argmax(Y_train.squeeze(1), dim=1)

        loss = criterion(output_probabilities, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch%100 == 0:
            print(f"LOSS: {loss.item()}")
        
print(f"Final loss: {loss.item()}\n\n")


with torch.no_grad():
    model.eval()
    predicted_labels = model.predict(X_test)


pred = idx_to_one_hot(predicted_labels, output_size=output_size)

Y_test = np.array(Y_test)

accuracy = accuracy_score(pred, Y_test)

print(f"\n Accuracy: {accuracy}\n")


