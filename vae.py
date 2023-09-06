import torch
import torch.nn as nn
from torch.utils.data import Dataset

import torch.nn.functional as F

import cv2
from torch.utils.data import Dataset

class MeDataset(Dataset):
    def __init__(self, a, b):
        self.custom_dataset = []
        for idx in range(a, b):
            image = cv2.imread(f"images/image_{str(idx).zfill(6)}.png")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)/255
            self.custom_dataset.append( (image, 0) )

    def __len__(self):
        return len(self.custom_dataset)
    
    def __getitem__(self, idx):
        image = self.custom_dataset[idx][0]
        label = torch.tensor(self.custom_dataset[idx][1])
        return (image, label)



class VAE(nn.Module):
    def __init__(self, imgChannels=3, featureDim=32*52*72, zDim=128):
        super(VAE, self).__init__()

        self.featureDim = featureDim
        self.encConv1 = nn.Conv2d(imgChannels, 16, 5)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 5)
        self.decConv2 = nn.ConvTranspose2d(16, imgChannels, 5)

    def encoder(self, x):
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = x.view(-1, 32*52*72)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 32, 52, 72)
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def get_z(self, x):
        mu, logVar = self.encoder(x)
        return self.reparameterize(mu, logVar)

    def forward(self, x):
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar

