#Load libraries
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
import torch
torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


# transforms
transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # 0 - 255 to 0 - 1 numpy to tensors
    transforms.Normalize([0.5, 0.5, 0.5], # 0-1 [-1, 1], formula x - mean / std
                         [0.5, 0.5, 0.5])
])

#dataloader
#Path for training and testing directory

train_path = "C:/Users/user/OneDrive/Desktop/jim_neural_network/face_classification_500/train_sample"
test_path = "C:/Users/user/OneDrive/Desktop/jim_neural_network/face_classification_500/dev_sample"

train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=transformer),
    batch_size=16, shuffle=True
)

test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=transformer),
    batch_size=16, shuffle=True
)

root = pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

num_classes = 500 

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
        
model = ConvNet(num_classes = 500).to(device)

# here we make use of Adam optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay = 0.001)
loss_function = nn.CrossEntropyLoss()

num_epochs = 50

train_count = len(glob.glob(train_path+'/**/*.jpg'))
test_count = len(glob.glob(test_path + '/**/*.jpg'))

print(train_count, test_count)


best_accuracy = 0.0

for epoch in range(50):

    #evaluation and training on training dataset
    model.train()
    # initial training accuracy
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        
        optimizer.zero_grad()
        
        outputs = model(images)        
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data*images.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    # Evaluation on testing dataset
    
    model.eval()

    test_accuracy = 0.0
    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy = test_accuracy / test_count
    
    
    print('Epoch: ' + str(epoch) + 'Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))
    
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_model.pth')
        best_accuracy = test_accuracy
    
