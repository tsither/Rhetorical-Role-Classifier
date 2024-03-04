import torch.nn as nn
import torch



def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
class CNN_BiLSTM(nn.Module):
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
                                   nn.Softmax(dim=1),
        )
        
        self.apply(init_weights)
        
    def forward(self, x):


        x = x.permute(0, 2, 1) 
        x = self.cnn(x)
        x = self.relu(x)  # Applying ReLU activation

        x = x.permute(0, 2, 1)  

        lstm_out, _ = self.bilstm(x)
        lstm_out = lstm_out[:, -1, :]

        out = self.dense(lstm_out)

        return out