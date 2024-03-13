import torch
import torch.nn as nn
import torch.nn.functional as F



class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, output_size)  

    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        output_probs = F.softmax(output, dim=1)  

        return output_probs
    
    def predict(self, x):
        with torch.no_grad():
            self.eval()
            output = self(x)
            _, predicted_classes = torch.max(output, 1)
            return predicted_classes