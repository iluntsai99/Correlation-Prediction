import os
from os import listdir
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import time
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

data_dir = Path("./correlation_assignment")
seed = 19991210
batch_size = 128
lr = 1e-4
num_epoch = 10

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def readfile(correlation):
    x = np.zeros((len(correlation), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(correlation)), dtype=np.float32)
    for i, path in enumerate(tqdm(correlation.items())):
        img = cv2.imread(str(data_dir)+"/images/"+path[0]+".png")
        x[i, :, :] = cv2.resize(img,(128, 128))
        y[i] = path[1]
        # print(path, y[i])
        # img = Image.fromarray(x[i])
        # img.save("test"+path)
    return x, y

def preprocess():
    # Read CSV
    print("Preprocessing...")
    with open(data_dir / "responses.csv", newline='') as csvfile:
        all_lines = csvfile.readlines()
    correlation = dict()
    for i in range(1, len(all_lines)):
        id, corr = all_lines[i].strip().split(',')
        correlation[id] = corr
    # all_id = [f for f in listdir(data_dir / "images")]
    X, Y = readfile(correlation)
    train_x, dev_x, train_y, dev_y = train_test_split(X, Y, test_size=0.1, random_state=seed)
    print("Train size: {}. Dev size: {} ".format(len(train_x), len(dev_x)))
    return train_x, dev_x, train_y, dev_y

transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
])

class ImgDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.y = torch.FloatTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.transform(self.x[index]), self.y[index]

class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),  # [6, 128, 128]
            nn.BatchNorm2d(8),
            nn.Tanh(),
            nn.MaxPool2d(4, 4, 0),  # [6, 32, 32]; Pooling for efficiency
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(8 * 32 * 32, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out = self.cnn_layers(x)  # Extract features by convolutional layers
        out = out.flatten(1)    # Flatten the feature map before going to fc layers
        return self.fc_layers(out)


def main(train_x, dev_x, train_y, dev_y):
    train_set = ImgDataset(train_x, train_y, transform)
    dev_set = ImgDataset(dev_x, dev_y, transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Regression().to(device)
    print("Total param: {}".format(sum(p.numel() for p in model.parameters())))
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_loss = 0.0
        dev_loss = 0.0

        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            if epoch == 0:
            	print("Note: skipping first training epoch in order to see the loss before training!")
                break
            optimizer.zero_grad()
            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].to(device).unsqueeze(1))
            # print(train_pred)
            # print(data[1])
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(dev_loader):
                dev_pred = model(data[0].cuda())
                batch_loss = loss(dev_pred, data[1].cuda().unsqueeze(1))
                dev_loss += batch_loss.item()
            
            print("Ground truth: {}, pred: {}".format(data[1][-1], dev_pred[-1]))
            print('[%03d/%03d] %2.2f sec(s) Train Loss: %3.6f | Val loss: %3.6f' % \
                (epoch + 1, num_epoch, time.time()-epoch_start_time, \
                train_loss/train_set.__len__(), dev_loss/dev_set.__len__()))

    #save model
    torch.save(model.state_dict(), 'model.pkl')

if __name__ == "__main__":
    same_seeds(seed)
    main(preprocess())