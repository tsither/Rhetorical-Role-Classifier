from typing import Tuple

import torch
import torch.nn as nn
from transformers import BertModel

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class CNN(nn.Module):
    def __init__(self, word_input_channels:int = 1,
                word_output_channels:int = 1,
                word_kernel_size:Tuple[int,int] = (5,1),
                sent_input_channels:int = 3,
                sent_output_channels:int = 1,
                dropout:float = 0.1,
                ) -> None:
        super().__init__()
        # Word Level CNN
        self.word_conv = nn.Sequential(nn.Conv2d(in_channels = word_input_channels,
                                                out_channels = word_output_channels,
                                                kernel_size = word_kernel_size),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size = (2,1)),
                                       nn.Dropout(p=dropout)
        )
        # Sentence Level    
        self.sent_conv = nn.Conv2d(in_channels = sent_input_channels,
                                   out_channels = sent_output_channels,
                                   kernel_size = (1,1))
        
        self.apply(init_weights)
        
    def forward(self,x):
        sent_ten = torch.Tensor()
        ind_x = x.unbind(0)
        for int_x in ind_x:
            int_x = int_x.unsqueeze(0)
            int_x = self.word_conv(int_x)
            sent_ten = torch.cat((sent_ten,int_x),dim=0)
        
        x = self.sent_conv(sent_ten)
        
        return x
        
class BiLSTM(nn.Module):
    def __init__(self,
                input_size:int = 768,
                hidden_size:int = 256,
                num_layers:int = 2,
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
        
        self.dense = nn.Sequential(nn.Linear(hidden_size*2, 128),
                                   nn.Dropout(p=dropout),
                                   nn.Linear(128, output_size),
                                   nn.Softmax(dim=1),
        )
        
        self.apply(init_weights)
        
    def forward(self, x):
        for i in range(x.size(-2)):
            if i == 0:
                out, (hidden,cell) = self.bilstm(x[:,i,:].unsqueeze(1))
            else:
                out, (hidden,cell) = self.bilstm(x[:,i,:].unsqueeze(1),(hidden,cell))

        # Take the output from the last time step
        out = out[:, -1, :]

        # Fully connected layers
        out = self.dense(out)

        return out
    

class CNN_BiLSTM(nn.Module):
    def __init__(self,
                word_input_channels:int = 1,
                word_output_channels:int = 1,
                word_kernel_size:Tuple[int,int] = (5,1),
                sent_input_channels:int = 3,
                sent_output_channels:int = 1,
                input_size:int = 768,
                hidden_size:int = 256,
                num_layers:int = 1,
                output_size:int = 13,
                dropout:float = 0.1
                ) -> None:
        super(CNN_BiLSTM,self).__init__()
        # Word Level CNN
        self.word_conv = nn.Sequential(nn.Conv2d(in_channels = word_input_channels,
                                                out_channels = word_output_channels,
                                                kernel_size = word_kernel_size),
                                       nn.MaxPool2d(kernel_size = (2,1)),
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout)
        )
        # Sentence Level    
        self.sent_conv = nn.Conv2d(in_channels = sent_input_channels,
                                   out_channels = sent_output_channels,
                                   kernel_size = (1,1))
        # BiLSTM
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.bilstm = nn.LSTM(input_size = input_size,
                              hidden_size = hidden_size,
                              num_layers = num_layers,
                              batch_first=True,bidirectional=True)
        
        self.dense = nn.Sequential(nn.Linear(in_features= hidden_size*2,out_features= 128),
                                   nn.ReLU(),
                                #    nn.Linear(in_features= 256, out_features= 128, bias=True),
                                #    nn.ReLU(),
                                   nn.Linear(in_features= 128, out_features= 64, bias=True),
                                   nn.ReLU(),
                                   nn.Linear(in_features= 64, out_features= output_size, bias=True),
                                   nn.Softmax(dim=0),
        )
        
        self.apply(init_weights) #pytorch weight initialization is poor by default
        
    def forward(self, x):
        sent_ten = torch.Tensor()
        # Passing through the word level CNN
        # Takes sentence word-level embeddings of 3 sentences
        # passes each sentence through the word level CNN 
        # concats the output to form a 3,38,768 tensor
        ind_x = x.unbind(0)
        for int_x in ind_x:
            int_x = int_x.unsqueeze(0)
            int_x = self.word_conv(int_x)
            sent_ten = torch.cat((sent_ten,int_x),dim=0)
            
        # 3,38,768 tensor passes through sentence level CNN
        # output is a 1,38,768 tensor
        x = self.sent_conv(sent_ten)
        
        # Forward pass through LSTM layer
        for i in range(x.size(-2)):
            if i == 0:
                out, (hidden,cell) = self.bilstm(x[:,i,:].unsqueeze(1))
            else:
                out, (hidden,cell) = self.bilstm(x[:,i,:].unsqueeze(1),(hidden,cell))
        
        # out, (hidden,cell) = self.bilstm(x, (hidden, cell))

        # Take the output from the last time step
        out = out.view(-1)

        # Fully connected layers
        out = self.dense(out)
        return out

    # def init_hidden(self, batch_size=1):
    #     """
    #     Initialize the hidden and cell states of the LSTM model.

    #     Args:
    #         batch_size (int, optional): Batch size for initialization. Defaults to 1.

    #     Returns:
    #         tuple: Tuple containing the initialized hidden and cell states.
    #     """
    #     hidden = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
    #     cell = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
    #     return hidden, cell
    