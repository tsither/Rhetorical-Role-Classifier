import torch.nn as nn

class CRF_BiLSTM(nn.Module):
    def __init__(self,
                input_size:int = 768,
                hidden_size:int = 128,
                num_layers:int = 1,
                output_size:int = 13,
                dropout:float = 0.1
                ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.bilstm = nn.LSTM(input_size = input_size,
                              hidden_size = hidden_size,
                              num_layers = num_layers,
                              bidirectional=True)
        
        self.dense = nn.Sequential(nn.Dropout(p=dropout),
                                   nn.Linear(hidden_size*2, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, output_size),
                                   
                                   
        )
        self.crf = CRF(num_tags=output_size,batch_first=True)
        
        
    def forward(self, x, labels):

        # print(f"LABELS: {labels.shape}")

        # print(f"X: {x.shape}")

        lstm_out, _ = self.bilstm(x)

        lstm_out = lstm_out[:, -1, :] 
        emissions = self.dense(lstm_out)


        # print(f"OUT: {emissions.shape}")

        # print(f"Labels: {labels.shape}")

        # emissions = out.unsqueeze(1)
        # print(f"OUT: {out.shape}")
        batch_size = emissions.size(0)
        emissions = emissions.unsqueeze(1).expand(batch_size, x.size(1), -1)

        labels = labels.unsqueeze(1)



        loss = -self.crf(emissions, labels)

        return loss
    
    def predict(self, x):
        """
        Predicts the most likely label sequence for a given input sequence.

        Args:
            x (torch.Tensor): Input sequence with shape (batch_size, sequence_length, input_size)

        Returns:
            torch.Tensor: Predicted label sequence with shape (batch_size, sequence_length)
        """
        
        self.eval()
        
        with torch.no_grad():
            lstm_out, _ = self.bilstm(x)
            lstm_out = lstm_out[:, -1, :]  
            emissions = self.dense(lstm_out)
            # print(f"emissions shape: {emissions.shape}")


        _, predictions = emissions.max(dim=-1)


        return predictions
        
        

##############################################################################################################################









def train_crf_model(model, data_loader, optimizer, epochs):
    model.train()
    batch_loss = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        for doc_idx in tqdm(range(246)):
                TRAIN_emb = data_loader(filepath=f"train_document/doc_{doc_idx}/embedding")
                TRAIN_labels = data_loader(filepath=f"train_document/doc_{doc_idx}/label")
                if TRAIN_emb.size(0) == 0:
                    continue

                #for BiLSTM-CRF model, the model already returns the loss in the forward function
                loss = model(TRAIN_emb, TRAIN_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Epoch: {epoch+1} | Document: {doc_idx+1}/246 | Loss: {loss.item():.5f}")

        batch_loss.append(loss.item())
    return np.mean(batch_loss)



def crf_train_test(parameters, model, data_loader):
    result = []
    model_opt = torch.optim.Adam(model.parameters(), lr= parameters['learning_rate'])
    loss_function = nn.CrossEntropyLoss()
    print("\nWorking with: ")
    print(parameters)
    print(f"Model type: {model.__class__.__name__}")
    print("Train type: default")
    print("\n")


    print(f'{"Starting Training":-^100}')
    train_loss = train_crf_model(model, data_loader, model_opt, parameters['epochs'])
    
    avg_accuracy, avg_f1= evaluate_crf_model(model, data_loader)
    
    print("Average accuracy: {}".format(avg_accuracy))
    print("Average F1: {}".format(avg_f1))
    
    result.append((parameters, (avg_accuracy,avg_f1)))


    return result

