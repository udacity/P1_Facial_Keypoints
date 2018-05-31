## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
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
        # just looks like vgg16
        self.conv1_1 = nn.Conv2d(1, 32, 3, padding=1)  # 96
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)  # 48
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)  # 24
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)  # 12
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)  # 6

        self.fc5 = nn.Linear(6*6*256, 4096)
        self.drop5 = nn.Dropout(0.5)

        self.fc6 = nn.Linear(4096, 4096)
        self.drop6 = nn.Dropout(0.5)

        self.fc7 = nn.Linear(4096, 68*2)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        #print('begin', x.size())
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        #print(x.size())
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        #print(x.size())
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        #print(x.size())
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        #print(x.size())
        x = self.pool4(x)

        #print(x.size())
        x = x.view((x.size(0), -1))
        #print(x.size())
        x = self.drop5(F.relu(self.fc5(x)))
        x = self.drop6(F.relu(self.fc6(x)))
        x = self.fc7(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
