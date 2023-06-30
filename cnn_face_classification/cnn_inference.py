import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
from torch.utils.data import DataLoader
import torch.functional as F
import torchvision
from io import open
import os
from PIL import Image
import pathlib
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
pred_path = "C:/Users/user/OneDrive/Desktop/jim_neural_network/face_classification_500/test_sample"
transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # 0 - 255 to 0 - 1 numpy to tensors
    transforms.Normalize([0.5, 0.5, 0.5], # 0-1 [-1, 1], formula x - mean / std
                         [0.5, 0.5, 0.5])
])
pred_loader=DataLoader(
    torchvision.datasets.ImageFolder(pred_path,transform=transformer),
    batch_size=16, shuffle=True
)
pred_count = len(glob.glob(pred_path + '/**/*.jpg'))

class ConvNet(nn.Module):
    def __init__(self, num_classes = 500):
        super(ConvNet, self).__init__()

        # output size after convolution filter
        #((w - f+ 2P/s) + 1
        
        # input shape = (256, 3, 224, 224)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1,padding=1)
        #Shape = (256, 12, 224, 224)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        #Shape = (256, 12, 224, 224)
        self.relu1 = nn.ReLU()
        #Shape = (256, 12, 224, 224)
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        #reduce the image size be factor 2
        #Shape = (256, 12, 112, 112)
      
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1,padding=1)      
        #Shape = (256, 20, 112, 112)            
        self.bn2 = nn.BatchNorm2d(num_features=20)
        #Shape = (256, 20, 112, 112)
        self.relu2 = nn.ReLU()
        #Shape = (256, 20, 112, 112)
        self.pool2 = nn.MaxPool2d(kernel_size = 2)
        #reduce the image size be factor 2
        #Shape = (256, 12, 56, 56)

        self.conv3 =nn.Conv2d(in_channels = 20, out_channels = 32, kernel_size = 3, stride=1, padding=1)
        #Shape = (256, 32, 56, 56)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        #Shape = (256, 32, 56, 56)
        self.relu3 = nn.ReLU()
        #Shape = (256, 32, 56, 56)
        self.pool3 = nn.MaxPool2d(kernel_size = 2)
        #Shape = (256, 32, 28, 28)

        # final fully connected layer
        self.fc1 = nn.Linear(in_features = 28*28*32, out_features=128)
        self.bn4 = nn.BatchNorm1d(num_features = 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features = 128, out_features=500)
     
        #Feed forwad function
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)            
        output=self.pool1(output)
            
        output=self.conv2(output)
        output=self.bn2(output)
        output=self.relu2(output)
        output=self.pool2(output)

        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
        output=self.pool3(output)

        # fully connected layer    
        #Above output will be in matrix form, with shape (256,32,112,112)           
        output=output.view(-1,32*28*28)
        output = self.fc1(output)
        output = self.bn4(output)
        output = self.relu4(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output

checkpoint=torch.load('best_model.pth')
model=ConvNet(num_classes=500)
model.load_state_dict(checkpoint)
model.eval()


# calculate prediction accuracy
pred_accuracy = 0.0

with torch.no_grad():
    for i, (images, labels) in enumerate(pred_loader):

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        pred_accuracy += int(torch.sum(prediction == labels.data))

    pred_accuracy = pred_accuracy / pred_count

print(pred_accuracy)