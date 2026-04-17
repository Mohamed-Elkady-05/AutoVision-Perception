import torch
import torch.nn as nn
import torch.nn.functional as F

class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficSignCNN, self).__init__()
        
        # 1. Convolutional Block 1
        # Input: 3 channels (RGB), Output: 32 feature maps
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2. Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3. Convolutional Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 4. Fully Connected Block
        # Assuming input images are resized to 32x32.
        # After three 2x2 pooling layers, the dimensions will be reduced to 4x4.
        # Flattened size: 128 channels * 4 * 4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        
        # Regularization: Dropout randomly zeroes 50% of the neurons to prevent overfitting
        self.dropout = nn.Dropout(p=0.5) 
        
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Pass through Conv -> ReLU -> Pool sequence
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 4 * 4)
        
        # Pass through Fully Connected layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x