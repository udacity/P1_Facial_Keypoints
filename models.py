## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # original image size is 224 x 224,  
        # so input 224 -> (Width-Filter size)/Stride +1 = (224-5)/1 + 1 = 220 is the output size
        # after maxpool it is 110
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)     #  (110-3)/1 + 1 = 108. after maxpool it is 108 / 2 = 54
        self.conv2_bn = nn.BatchNorm2d(64)        
        self.conv3 = nn.Conv2d(64, 128, 3)    # (54-3)/1 + 1 = 52. after maxpool it is 52 / 2 = 26
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)   # (26-3)/1 + 1 = 24. after maxpool it is 24 / 2 = 12
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 1)   # (12-1)/1 + 1 = 12. after maxpool it is 12 / 2 = 6
        self.conv5_bn = nn.BatchNorm2d(512)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.maxPool = nn.MaxPool2d(2, 2)         

        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 136)        
  
        self.drop = nn.Dropout(p=0.3)      
        
        
    # define the feedforward behavior
    def forward(self, x):
        # two conv/relu + pool layers
        x = self.maxPool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.maxPool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.maxPool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.maxPool(F.relu(self.conv4_bn(self.conv4(x))))
        x = self.maxPool(F.relu(self.conv5_bn(self.conv5(x))))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        
        # final output
        return x

    
class NaimishNet(nn.Module):

    def __init__(self):
        super(NaimishNet, self).__init__()

        # original image size is 224 x 224,
        # so input 224 -> (Width-Filter size)/Stride +1 = (224-4)/1 + 1 = 221 is the output size
        # after maxpool it is 110
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)     # (110-3)/1 + 1 = 108. after maxpool it is 108 / 2 = 54
        self.conv3 = nn.Conv2d(64, 128, 2)    # (54-2)/1 + 1 = 53. after maxpool it is 53 / 2 = 26
        self.conv4 = nn.Conv2d(128, 256, 1)   # (26-1)/1 + 1 = 26. after maxpool it is 26 / 2 = 13

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.maxPool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256*13*13, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 136)        

        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.6)

    # define the feedforward behavior
    def forward(self, x):
        # two conv/relu + pool layers
        x = self.drop1(self.maxPool(F.relu(self.conv1(x))))
        x = self.drop2(self.maxPool(F.relu(self.conv2(x))))
        x = self.drop3(self.maxPool(F.relu(self.conv3(x))))
        x = self.drop4(self.maxPool(F.relu(self.conv4(x))))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.drop5(x)
        x = F.relu(self.fc2(x))
        x = self.drop6(x)
        x = self.fc3(x)

        # final output
        return x
    

class BorrowedNet(nn.Module):

    def __init__(self):

        super(BorrowedNet, self).__init__()

        # Covolutional Layers
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 2)

        # Maxpooling Layer
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features = 36864, out_features = 1000) # The number of input gained by "print("Flatten size: ", x.shape)" in below
        self.fc2 = nn.Linear(in_features = 1000,    out_features = 1000)
        self.fc3 = nn.Linear(in_features = 1000,    out_features = 136) # the output 136 in order to having 2 for each of the 68 keypoint (x, y) pairs

        # Dropouts
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        self.drop6 = nn.Dropout(p = 0.6)

    def forward(self, x):

        # First - Convolution + Activation + Pooling + Dropout
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop1(x)
        #print("First size: ", x.shape)

        # Second - Convolution + Activation + Pooling + Dropout
        x = self.drop2(self.pool(F.relu(self.conv2(x))))
        #print("Second size: ", x.shape)

        # Third - Convolution + Activation + Pooling + Dropout
        x = self.drop3(self.pool(F.relu(self.conv3(x))))
        #print("Third size: ", x.shape)

        # Forth - Convolution + Activation + Pooling + Dropout
        x = self.drop4(self.pool(F.relu(self.conv4(x))))
        #print("Forth size: ", x.shape)

        # Flattening the layer
        x = x.view(x.size(0), -1)
        #print("Flatten size: ", x.shape)

        # First - Dense + Activation + Dropout
        x = self.drop5(F.relu(self.fc1(x)))
        #print("First dense size: ", x.shape)

        # Second - Dense + Activation + Dropout
        x = self.drop6(F.relu(self.fc2(x)))
        #print("Second dense size: ", x.shape)

        # Final Dense Layer
        x = self.fc3(x)
        #print("Final dense size: ", x.shape)

        return x