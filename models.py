import torch.nn as nn
from utils import init_weights
#from torchcrf import CRF
import torch

###########################################################################

# file containing the two current models 

###########################################################################




class BiLSTM(nn.Module):
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
        
        self.apply(init_weights)
        
    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        lstm_out = lstm_out[:, -1, :]

        out = self.dense(lstm_out)

        return out
        

class CNN_BiLSTM(nn.Module):
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


        self.cnn = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        
        self.bilstm = nn.LSTM(input_size = hidden_size,
                              hidden_size = hidden_size,
                              num_layers = num_layers,
                              bidirectional=True)
        
        self.dense = nn.Sequential(nn.Dropout(p=dropout),
                                   nn.Linear(hidden_size*2, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, output_size),
        )
        
        self.apply(init_weights)
        
    def forward(self, x):


        x = x.permute(0, 2, 1) 
        x = self.cnn(x)
        x = self.relu(x)  

        x = x.permute(0, 2, 1)  

        lstm_out, _ = self.bilstm(x)
        lstm_out = lstm_out[:, -1, :]

        out = self.dense(lstm_out)

        return out
