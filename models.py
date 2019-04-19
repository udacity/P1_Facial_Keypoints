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
        # input image is 224x224
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #output of first is (W-F)/S + 1 = (224-5)/1 +1 = 220 
        self.conv1 = nn.Conv2d(1, 32, 4)
        #after one maxpool it becomes 110
        self.pool = nn.MaxPool2d(2, 2)
        #next conv layer
        #output of scnd is (W-F)/S + 1 = (110-5)/1 +1 = 106
        self.conv2 = nn.Conv2d(32, 64, 3)
        #another pooling layer down to 55
        self.conv3 = nn.Conv2d(64, 128, 2)
        # 32 outputs * the 55*55 filtered/pooled map size
        #self.fc1 = nn.Linear(128*24*24, 1000)
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.fc1 = nn.Linear(256*13*13, 1000)
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.1)
        self.fc2_drop = nn.Dropout(p=0.2)
        self.fc3_drop = nn.Dropout(p=0.3)
        self.fc4_drop = nn.Dropout(p=0.4)
        self.fc5_drop = nn.Dropout(p=0.5)
        self.fc6_drop = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
                # two conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.fc1_drop(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.fc2_drop(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.fc3_drop(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.fc4_drop(x)
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc5_drop(x)
        x = self.fc2(x)
        x = self.fc6_drop(x)
        x = self.fc2(x)
        x = self.fc6_drop(x)
        x = self.fc3(x)
        # final output
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
