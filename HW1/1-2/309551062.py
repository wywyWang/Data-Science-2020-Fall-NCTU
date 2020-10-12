import sys
from PIL import Image
from tqdm import tqdm
import numpy as np
import utils

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


resize_size = (224, 224)
batch_size = 32
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
}

if __name__ == '__main__':
    filename = sys.argv[1]
    testset = utils.customDataset(datatype='test', transform=data_transforms['test'], filename=filename)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load models
    PATH = './model/efficientnet-b4_20201011-232220_0.001_0_0.45149814031926444_0.4434005710601455'
    new_net = utils.Net()
    new_net.load_state_dict(torch.load(PATH))
    new_net.to(device)
    new_net.eval()
    
    # inference
    y_pred_test = np.array([])
    with torch.no_grad():
        for data_test in tqdm(testloader):
            images = data_test.to(device)
            outputs = new_net(images)

            _, y_pred_tag = torch.max(outputs, 1)
            y_pred_test = np.hstack([y_pred_test, y_pred_tag.cpu().detach().numpy()])

    output_filename = './classification.txt'
    with open(output_filename, 'w') as f_test:
        for each_predict in y_pred_test:
            f_test.write(str(int(each_predict)))