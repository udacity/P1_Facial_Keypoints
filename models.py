## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

'''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel      
        self.conv1 = nn.Conv2d(1, 32, 5)
        # original image size is 224 x 224,  
        # so input 224 -> (Width-Filter size)/Stride +1 = (224-5)/1 + 1 = 220 is the output size
        # after maxpool it is 110

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.maxPool = nn.MaxPool2d(2, 2)         
        
        self.conv2 = nn.Conv2d(32, 64, 3)     #  (110-3)/1 + 1 = 108. after maxpool it is 108 / 2 = 54
        #self.conv3 = nn.Conv2d(64, 128, 3)    # (54-3)/1 + 1 = 52. after maxpool it is 52 / 2 = 26
        #self.conv4 = nn.Conv2d(128, 256, 3)   # (26-3)/1 + 1 = 24. after maxpool it is 24 / 2 = 12
        #self.conv5 = nn.Conv2d(256, 512, 1)   # (12-1)/1 + 1 = 12. after maxpool it is 12 / 2 = 6

        self.fc1 = nn.Linear(64 * 54 * 54, 1024)
        self.fc2 = nn.Linear(1024, 136)        
        
        # self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        # self.fc2 = nn.Linear(1024, 136)        

        
        # self.drop = nn.Dropout(p=0.3)      
        
        
    # define the feedforward behavior
    def forward(self, x):
        # two conv/relu + pool layers
        x = self.maxPool(F.relu(self.conv1(x)))
        x = self.maxPool(F.relu(self.conv2(x)))
        # x = self.maxPool(F.relu(self.conv3(x)))
        # x = self.maxPool(F.relu(self.conv4(x)))
        # x = self.maxPool(F.relu(self.conv5(x)))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
#        x = self.drop(x)
        x = self.fc2(x)
        
        # final output
        return x
'''    

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # original image size is 224 x 224,  
        # so input 224 -> (Width-Filter size)/Stride +1 = (224-5)/1 + 1 = 220 is the output size
        # after maxpool it is 110
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)     #  (110-3)/1 + 1 = 108. after maxpool it is 108 / 2 = 54
        self.conv3 = nn.Conv2d(64, 128, 3)    # (54-3)/1 + 1 = 52. after maxpool it is 52 / 2 = 26
        self.conv4 = nn.Conv2d(128, 256, 3)   # (26-3)/1 + 1 = 24. after maxpool it is 24 / 2 = 12
        self.conv5 = nn.Conv2d(256, 512, 1)   # (12-1)/1 + 1 = 12. after maxpool it is 12 / 2 = 6

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.maxPool = nn.MaxPool2d(2, 2)         

        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 136)        
  
        self.drop = nn.Dropout(p=0.3)      
        
        
    # define the feedforward behavior
    def forward(self, x):
        # two conv/relu + pool layers
        x = self.maxPool(F.relu(self.conv1(x)))
        x = self.maxPool(F.relu(self.conv2(x)))
        x = self.maxPool(F.relu(self.conv3(x)))
        x = self.maxPool(F.relu(self.conv4(x)))
        x = self.maxPool(F.relu(self.conv5(x)))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        
        # final output
        return x
