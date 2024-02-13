import torch.nn as nn

class word_level_CNN(nn.Module):
    def __init__(self, input_channels : int = 60, output_channels : int = 30, kernel_size : int = (5,1)):
        super(word_level_CNN, self).__init__()
        # Convolutional layer
        self.conv1 = nn.Conv2d(in_channels = input_channels,
                               out_channels = output_channels,
                               kernel_size = kernel_size)
        # ReLU activation
        self.relu = nn.ReLU()
        # Max pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size = (2,1))
        
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = self.conv1(x)
        # print("After Convolution : ", x.size())
        x = self.relu(x)
        x = self.max_pool(x)
        # print("After Max-Pooling : ", x.size())
        return x
    
class sent_level_CNN(nn.Module):
    def __init__(self, input_channels : int = 3, output_classes : int = 1):
        super(sent_level_CNN, self).__init__()
        # Convolutional layer
        self.conv1 = nn.Conv2d(in_channels = input_channels,
                               out_channels = output_classes,
                               kernel_size = (1,1))
    def forward(self, x):
        # Forward pass through the layers
        x = self.conv1(x)
        # print("After Convolution : ", x.size())
        
        return x