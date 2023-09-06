import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random

import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from vae import VAE, MeDataset


batch_size = 128
learning_rate = 1e-3
num_epochs = 200


train_loader = torch.utils.data.DataLoader(MeDataset(    0,19000), batch_size=batch_size, shuffle=True)
val_loader =   torch.utils.data.DataLoader(MeDataset(19000,19500), batch_size=batch_size)
test_loader =  torch.utils.data.DataLoader(MeDataset(19500,20000), batch_size=1)

print('Dataset loaded')

net = VAE().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    batch_loss = []
    samples = 0
    for idx, data in enumerate(train_loader, 0):
        imgs, _ = data
        samples += imgs.shape[0]
        imgs = imgs.to(device)
        out, mu, logVar = net(imgs)
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
    train_loss = np.absolute(np.array(batch_loss)).sum()/samples

    with torch.no_grad():
        batch_loss = []
        samples = 0
        for idx, data in enumerate(val_loader, 0):
            imgs, _ = data
            samples += imgs.shape[0]
            imgs = imgs.to(device)
            out, mu, logVar = net(imgs)
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence
            batch_loss.append(loss.item())
        val_loss = np.absolute(np.array(batch_loss)).sum()/samples
    
    print(f'Epoch {str(epoch).zfill(4)}: train({str(train_loss).rjust(12)}) val({str(val_loss).rjust(12)})')

    torch.save(net.state_dict(), f"vae_{str(epoch).zfill(3)}.pth")


net.eval()
i = 0
with torch.no_grad():
    for idx, data in enumerate(test_loader, 0):
        imgs, _ = data
        imgs = imgs.to(device)
        img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
        plt.subplot(121)
        image_to_show = np.squeeze(img*255).astype(np.uint8)
        print(image_to_show.min(), image_to_show.max(), image_to_show.shape)
        # plt.imshow(image_to_show)
        out, mu, logVAR = net(imgs)
        outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])*255
        outimg = outimg.astype(np.uint8)
        print(outimg.min(), outimg.max(), outimg.shape)
        # plt.subplot(122)
        # plt.imshow(np.squeeze(outimg))
        concatenate = np.concatenate((image_to_show, outimg), axis=1)
        concatenate = cv2.cvtColor(concatenate, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"vision_{str(i).zfill(6)}.png", concatenate)
        i+= 1




