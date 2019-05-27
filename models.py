## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        self.pool = nn.MaxPool2d(2,2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(12*12*256, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        
#         self.drop1 = nn.Dropout(0.1)
#         self.drop2 = nn.Dropout(0.2)
#         self.drop3 = nn.Dropout(0.3)
#         self.drop4 = nn.Dropout(0.4)
        self.drop5 = nn.Dropout(0.25)
        self.drop6 = nn.Dropout(0.25)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.drop1(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.drop2(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.drop3(x)
        
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
#         x = self.drop4(x)
        
        flat = x.view(x.size(0), -1)
        
        out = self.drop5(F.relu(self.fc1(flat)))
        out = self.drop6(F.relu(self.fc2(out)))
        out = self.fc3(out)
        
        return out
