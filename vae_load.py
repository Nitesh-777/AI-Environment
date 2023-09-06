import sys
import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import random

import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from vae import VAE, MeDataset

batch_size = 128
learning_rate = 1e-4
num_epochs = 50

# train_loader = torch.utils.data.DataLoader(MeDataset(    0,19000), batch_size=batch_size, shuffle=True)
# val_loader =   torch.utils.data.DataLoader(MeDataset(19500,20000), batch_size=batch_size)
test_loader =  torch.utils.data.DataLoader(MeDataset(19000,19500), batch_size=1)

print('Dataset loaded')

net = VAE().to(device)
net.load_state_dict(torch.load(sys.argv[1]))

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
        plt.imshow(image_to_show)
        out, mu, logVAR = net(imgs)
        outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])*255
        outimg = outimg.astype(np.uint8)
        print(outimg.min(), outimg.max(), outimg.shape)
        plt.subplot(122)
        plt.imshow(np.squeeze(outimg))
        plt.show()
        i+= 1




