# %%
import os
import glob
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision.models import vgg19
from PIL import Image
from sklearn.model_selection import train_test_split
import random
import shutil
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor
import cv2

# %%
class ImageDataset(Dataset):
    def _init_(self,hr,lr, scale):
        self.hr_images = datasets.ImageFolder(root = hr)
        self.lr_images = datasets.ImageFolder(root = lr)
        self.scale = scale
    def _len_(self):
        return len(self.hr_images)
    def get_box(self, lr, hr, box_size=48):
        lr_height, lr_width = lr.shape[:2]
        hr_patch_size = box_size * self.scale
        begin = 0
        x_val = random.randrange(begin, lr_width - box_size + 1)
        y_val = random.randrange(begin, lr_height - box_size + 1)
        x_scaled= self.scale * x_val
        y_scaled = self.scale * y_val
        lr_val = lr[y_val : y_val + box_size, x_val : x_val + box_size, :]
        hr_val = hr[y_scaled:y_scaled + hr_patch_size,x_scaled:x_scaled + hr_patch_size, :]
        r = [lr_val, hr_val]
        return r
    
    def _getitem_(self, i):
        lowres, _ = self.lr_images[i]
        highres, _ = self.hr_images[i]
        lowres_image, highres_image = np.array(lowres), np.array(highres)
        lowres_image, highres_image = lowres_image / 255, highres_image / 255
        lowres_image, highres_image = self.get_box(lowres_image, highres_image)
        lowres_image, highres_image = np.transpose(lowres_image, (2, 0, 1)), np.transpose(highres_image, (2, 0, 1))
        return {'lr': lowres_image, 'hr': highres_image}

# %%
batch_size = 32
HRdataset_path = "./Datasets/"
HRtrain_path = HRdataset_path + "Train/"
HRval_path = HRdataset_path + "Val/"
HRtest_path = HRdataset_path + "Test/"
LRdataset_path = "./LRDatasets/"
LRtrain_path = LRdataset_path + "Train"
LRval_path = LRdataset_path + "Val"
LRtest_path = LRdataset_path + "Test"

# %%
train_dataloader = DataLoader(ImageDataset("./Datasets/", "./LRDatasets/",2), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dataloader = DataLoader(ImageDataset("./TestDatasets/", "./LRTestDatasets/",2), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# %%
import torch.nn.functional as F

class FSRCNN(nn.Module):
    def _init_(self, upscale_factor, num_channels=3): 
        super(FSRCNN, self)._init_()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, 56, kernel_size=5, padding=2),
            nn.PReLU()
        )

        self.shrinking = nn.Sequential(
            nn.Conv2d(56, 12, kernel_size=1),
            nn.PReLU()
        )

        self.non_linear_mapping = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.expanding = nn.Sequential(
            nn.Conv2d(12, 56, kernel_size=1),
            nn.PReLU()
        )

        self.deconvolution = nn.ConvTranspose2d(56, num_channels, kernel_size=9, stride=1, padding=4, output_padding=0)
        self.upscale_factor = upscale_factor

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.non_linear_mapping(x)
        x = self.expanding(x)
        x = F.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=True)
        x = self.deconvolution(x)
        return x


# %%


# %%
def train(model, train_dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        lr = data['lr']
        hr = data['hr']
        lr.to(device)
        hr.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_dataloader)

# %%
def validate(model, test_dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    psnr_running = 0.0
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            lr = data['lr']
            hr = data['hr']
            lr.to(device)
            hr.to(device)
            sr = model(lr.double())
            loss = criterion(sr, hr)
            running_loss += loss.item()
    return running_loss / len(test_dataloader), psnr_running / len(test_dataloader)

# %%
batch_size = 32
epochs = 15
lr = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FSRCNN(upscale_factor=2).double()
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
train_loss_epoch = []
val_loss_epoch = []
psnr_epoch = []
for epoch in range(1, epochs+1):
    train_loss = train(model, train_dataloader, criterion, optimizer, device)
    val_loss, psnr = validate(model, val_dataloader, criterion, device)
    print('Epoch [{}/{}], train_loss: {}, val_loss: {}, average psnr: {}'.format(epoch, epochs, train_loss, val_loss, psnr))
    train_loss_epoch.append(train_loss)
    val_loss_epoch.append(val_loss)
    psnr_epoch.append(psnr)

# %%
plt.plot(range(1, len(train_loss_epoch) + 1), train_loss_epoch, label='Training Loss')
plt.plot(range(1, len(val_loss_epoch) + 1), val_loss_epoch, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

# %%
import cv2
import numpy as np

psnr_values = []
for i, batch in enumerate(val_dataloader):
    imgs_lr = batch["lr"].cpu()
    imgs_hr = batch["hr"].cpu()
    sr_hr = model(imgs_lr)
    for j in range(imgs_hr.shape[0]):
        psnr = calc_psnr(sr_hr, imgs_hr)
        psnr_values.append(psnr)

average_psnr = sum(psnr_values) / len(psnr_values)
print("Average PSNR:", average_psnr)