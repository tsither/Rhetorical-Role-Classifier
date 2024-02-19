from typing import Tuple

import torch
import torch.nn as nn

class CNN_BiLSTM(nn.Module):
    def __init__(self,
                word_input_channels:int = 1,
                word_output_channels:int = 1, 
                word_kernel_size:Tuple[int,int] = (5,1),
                sent_input_channels:int = 3,
                sent_output_channels:int = 1,
                input_size:int = 768,
                hidden_size:int = 256,
                num_layers:int = 2,
                output_size:int = 13,
                ) -> None:
        super(CNN_BiLSTM,self).__init__()
        # Word Level CNN
        self.word_conv = nn.Conv2d(in_channels = word_input_channels,
                                   out_channels = word_output_channels,
                                   kernel_size = word_kernel_size)
        self.word_max_pool = nn.MaxPool2d(kernel_size = (2,1))
        # Sentence Level CNN
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
        self.fc1 = nn.Linear(hidden_size*2, 128)
        self.fc2 = nn.Linear(128, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, hidden, cell):
        sent_ten = torch.Tensor()
        # Passing through the word level CNN
        # Takes sentence word-level embeddings of 3 sentences
        # passes each sentence through the word level CNN 
        # concats the output to form a 3,38,768 tensor
        ind_x = x.unbind(0) 
        for int_x in ind_x:
            int_x = int_x.unsqueeze(0)
            print(int_x.size())
            # int_x = int_x.view(int_x.size(0),int_x.size(1), -1)
            int_x = self.word_conv(int_x)
            int_x = self.relu(int_x)
            int_x = self.word_max_pool(int_x)
            sent_ten = torch.cat((sent_ten,int_x),dim=0)
            
        # 3,38,768 tensor passes through sentence level CNN
        # output is a 1,38,768 tensor
        x = self.sent_conv(sent_ten)
        
        # Forward pass through LSTM layer
        out, (hidden,cell) = self.bilstm(x, (hidden, cell))

        # Take the output from the last time step
        out = out[:, -1, :]

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)

        return out, (hidden,cell)

    def init_hidden(self, batch_size=1):
        """
        Initialize the hidden and cell states of the LSTM model.

        Args:
            batch_size (int, optional): Batch size for initialization. Defaults to 1.

        Returns:
            tuple: Tuple containing the initialized hidden and cell states.
        """
        hidden = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
        cell = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
        return hidden, cell
    