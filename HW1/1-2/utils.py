from PIL import Image
from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.models as models


class customDataset(Dataset):
    def __init__(self, datatype, transform, filename):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
        self.transform = transform

        filename = open(filename, 'r')
        info = filename.readlines()
        self.images = [row.strip() for row in info]

        print("image shape: {}".format(len(self.images)))
        
        
    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        
        image = Image.open(self.images[index]).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        return image
        
    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.images)


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.pretrained = EfficientNet.from_pretrained('efficientnet-b4')

        # freeze pretrained models
        ct = 0
        for child in self.pretrained.children():
            # print()
            # print(child)
            ct += 1
            # if ct < 30:
            for param in child.parameters():
                param.requires_grad = False
        self.layer_cnt = ct

        self.relu0 = nn.ReLU()

        # self.fc1 = nn.Linear(1000, 512)
        # self.relu1 = nn.ReLU()
        # self.drop1 = nn.Dropout(p=0.2)

        # self.fc2 = nn.Linear(512, 2)

        self.fc = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.pretrained(x)
        x = self.relu0(x)

        # x = self.fc1(x)
        # x = self.relu1(x)
        # x = self.drop1(x)

        # x = self.fc2(x)

        x = self.fc(x)
        return x