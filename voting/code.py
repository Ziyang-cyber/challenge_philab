from torchensemble import VotingClassifier  # voting is a classic ensemble strategy

import io
import os
import torch
import zipfile
import requests
import torch.nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import satlaspretrain_models
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import rasterio
import tempfile
from typing import Optional
from lightning.pytorch import Trainer
from torch.utils.data import random_split
import satlaspretrain_models
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import copy
from rasterio.errors import NotGeoreferencedWarning
import warnings
import torch.nn as nn
from sklearn.metrics import f1_score
from torchvision.transforms import v2
from kernels import create_kernel
import multiprocessing
from model_foundation_local_rev2 import Foundation as Foundation_local
from blocks import CNNBlock
from torch import nn
from sklearn.metrics import f1_score
from kernels import create_kernel
from torchvision.transforms import v2
from torch.utils.data import Subset
from collections import Counter

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
weights_manager = satlaspretrain_models.Weights()

class SpatialHOT(nn.Module):
    """
    This loss allows the targets for the cross entropy loss to be multi-label.
    The labels are smoothed by a spatial gaussian kernel before being normalized.

    Input is expected to be of shape [B, C, H, W] and target is expected to be class integers of the shape [B, 1, H, W].

    Parameters
    ----------
    method : str, optional
        The method to use for smoothing the labels. One of 'half', 'max', or None.
        By setting a method, you ensure that the center pixel will never 'flip' to another class.
        Default is 'half'.
        - `'half'` will set the pixel to at least half the sum of the weighted classes for the pixel.
        - `'kernel_half'` will set the center of the kernel to be weighted as half the sum of the surrounding weighted classes. Multiplied by `(1 + self.kernel_size) / self.kernel_size`.
        - `'max'` will set the center of the kernel to be weighted as the maximum of the surrounding weighted classes. Multiplied by `(1 + self.kernel_size) / self.kernel_size`. 
        - `None` will not treat the center pixel differently. Does not ensure that the center pixel will not 'flip' to another class.

    classes : list[int]
        A sorted (ascending) list of the classes to use for the loss.
        Should correspond to the classes in the target. I.e. for ESA WorldCover it should be
        `[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]`.

    kernel_radius : float, optional
        The radius of the gaussian kernel. Default is 1.0. Can be fractional.

    kernel_circular : bool, optional
        Whether to use a circular kernel or not. Default is True.

    kernel_sigma : float, optional
        The sigma of the gaussian kernel. Default is 2.0. Can be fractional.

    epsilon : float, optional
        A small value to add to the denominator to avoid division by zero. Default is 1e-07.

    device : str, optional
        The device to use for the computations. Default is 'cuda' if available, else 'cpu'.

    channel_last : bool, optional
        Whether the input is channel last or not. Default is False.
    """
    def __init__(
        self,
        method: Optional[str] = "half",
        classes: Optional[list[int]] = None,
        kernel_radius: float = 1.0,
        kernel_circular: bool = True,
        kernel_sigma: float = 2.0,
        epsilon: float = 1e-07,
        device: Optional[str] = None,
        channel_last: bool = False,
    ) -> None:
        super().__init__()
        assert method in ["half", "kernel_half", "max", None], \
            "method must be one of 'half', 'kernel_half', 'max', or None"
        assert isinstance(classes, list) and len(classes) > 1, "classes must be a list of at least two ints"
        assert classes == sorted(classes), "classes must be sorted in ascending order"

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.method = method
        self.channel_last = channel_last
        self.eps = torch.Tensor([epsilon]).to(self.device)

        # Precalculate the classes and reshape them for broadcasting
        self.classes_count = len(classes)
        self.classes = torch.Tensor(classes).view(1, -1, 1, 1).to(self.device)

        # Calculate the kernel using numpy. It is slower, but it is only done once.
        self.kernel_np = create_kernel(
            radius=kernel_radius,
            circular=kernel_circular,
            distance_weighted=True,
            hole=False if method in [None, "max"] else True,
            method=3,
            sigma=kernel_sigma,
            normalised=False
        )

        # To ensure that the classes are not tied in probability, we multiply the center of the kernel by a factor
        # E.g. if the kernel is 5x5, then the center pixel will be multiplied by 25 / (25 - 1) = 1.04
        self.kernel_size = self.kernel_np.size
        self.kernel_width = self.kernel_np.shape[1]
        self.strength = self.kernel_size / (self.kernel_size - 1.0)

        # Set the center of the kernel to be at least half the sum of the surrounding weighted classes
        if method == "kernel_half":
            self.kernel_np[self.kernel_np.shape[0] // 2, self.kernel_np.shape[1] // 2] = self.kernel_np.sum() * self.strength

        # Padding for conv2d
        self.padding = (self.kernel_np.shape[0] - 1) // 2

        # Turn the 2D kernel into a 4D kernel for conv2d
        self.kernel = torch.Tensor(self.kernel_np).unsqueeze(0).repeat(self.classes_count, 1, 1, 1).to(device)


    def forward(self, target: torch.Tensor) -> torch.Tensor:
        """ Expects input to be of shape [B, C, H, W] or [B, H, W, C] if channel_last is True. """
        if self.channel_last:
            target = target.permute(0, 3, 1, 2)

        target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(self.device)

        # One-hot encode the target and cast to float
        target_hot = (target_tensor == self.classes).float()

        # Pad the targets using the same padding as the kernel and 'same' padding
        target_hot_pad = F.pad(target_hot, pad=(self.padding, self.padding, self.padding, self.padding), mode="replicate")

        # Convolve the padded targets with the custom kernel
        convolved = F.conv2d(
            target_hot_pad,
            weight=self.kernel,
            bias=None,
            padding=self.padding,
            groups=self.classes_count,
        )

        # Handle the flip protection methods
        if self.method == "max":
            valmax, argmax = torch.max(convolved, dim=1, keepdim=True)
            _, argmax_hot = torch.max(target_hot_pad, dim=1, keepdim=True)

            weight = torch.where(argmax == argmax_hot, convolved, valmax * self.strength)
            convolved = torch.where(target_hot_pad == 1, weight, convolved)

        elif self.method == "half":
            weight = self.kernel_np.sum() * self.strength
            convolved = torch.where(target_hot_pad == 1, weight, convolved)

        # No handling necessary for these two methods
        elif self.method == "kernel_half" or self.method is None:
            pass

        # Remove the padding
        convolved = convolved[:, :, self.padding:-self.padding, self.padding:-self.padding]

        # The output is not normalised, so we can either apply softmax or normalise it using the sum.
        target_soft = convolved / torch.maximum(convolved.sum(dim=(1), keepdim=True), self.eps)

        return target_soft.squeeze()



class Sentinel2Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.mean = np.array([396.46749898, 494.62101716, 822.31995507, 973.67493873, 2090.1119281, 1964.05106209, 1351.27389706, 102.73961525, 141.80384498, 300.74131944, 35.10253906, 9.7533334]).reshape(-1, 1, 1)
        self.std = np.array([145.55178973, 182.14915237, 237.00658701, 315.7291156, 692.95828227, 746.64353554, 572.45327819, 25.64108775, 59.59076925, 23.15898661, 15.39840718, 16.72491096]).reshape(-1, 1, 1)
        self.min = np.array([-1.393e+03, -1.169e+03, -7.220e+02, -6.840e+02, -4.120e+02, -3.350e+02, -2.580e+02, 6.400e+01, -9.999e+03, 8.000e+00, 1.000e+01, 0.000e+00]).reshape(-1, 1, 1)
        self.max = np.array([ 6568,  9659, 11368, 12041, 15841, 15252, 14647, 255, 4245, 4287, 100, 111]).reshape(-1, 1, 1)
        self.images = os.listdir(image_dir)

        self.hot_encoder = SpatialHOT(
            classes=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
            device="cuda",
        )

        print(" INiotted")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        label_name = os.path.join(self.label_dir, self.images[idx].replace('.tif', '.png'))

        # Read Sentinel-2 TIFF image (first 7 bands)
        with rasterio.open(img_name) as img_file:
            img = img_file.read([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # Reading all 12 bands
            img_spectral = img[0:7]
            img_spectral = img_spectral/10000  # Clip values to 0-10000 range and normalize to 0-1
            img_static = img[[7,8,9,11],:,:]

            img_static = (img_static - self.min[[7,8,9,11],:,:]) / (self.max[[7,8,9,11],:,:] - self.min[[7,8,9,11],:,:])
            img_static_lc = self.hot_encoder(img[10]).cpu().numpy()


            
        img = np.concatenate([img_spectral, img_static, img_static_lc], axis=0)

        # Read label PNG image
        label = np.array(Image.open(label_name))

        # Convert label to binary format if necessary
        label = (label > 0).astype(np.float32)

        # Convert arrays to PyTorch tensors
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).float()  # Add channel dimension to label

        if self.transform:
            image_and_mask = torch.cat([img, label.unsqueeze(0)], dim=0) 
            
            # img,label = self.transform(img,label)
            transformed = self.transform(image_and_mask)
            img = transformed[:22]
            label = transformed[22]



        # Upsample image and label to 512x512
        img = F.interpolate(img.unsqueeze(0), size=(512, 512), mode='bilinear').squeeze(0)
 

        # Convert label back to long tensor (for categorical data, if necessary)
        num_classes = 2
        label = label.long()
        label = F.one_hot(label, num_classes)  # Convert to one-hot encoding
        label = label.permute(2, 0, 1).float() # Reshape to (C, H, W) format
        return img, label


    
class FloodPredictorSpectral(nn.Module):
    def __init__(
        self,
        input_dim=7,
        output_dim=32,
        path_weights="",
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.path_weights = path_weights

        self.foundation = Foundation_local(
            input_dim=10, # B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
            depths=[3, 3, 4, 4, 5],  # 128, 64, 32, 16, 8
            dims=[32, 32, 64, 64, 128],
            img_size=128,
            latent_dim=1024,
            dropout=None,
            activation=nn.GELU(),
        )

        if self.path_weights != "":
            weights = torch.load(self.path_weights)


            
            self.foundation.load_state_dict(weights, strict=False)

        self.encoder = self.foundation.encoder # expects dim 32 input
        self.decoder = self.foundation.decoder

        self.stem = CNNBlock(self.input_dim, 32, activation=nn.GELU(), residual=False, chw=[self.input_dim, 128, 128])

    def forward(self, x):
        x = self.stem(x)
        embeddings, embeddings_cnn, skips, predictions = self.encoder(x)
        decoded = self.decoder(embeddings, skips)

        return decoded


class FloodPredictorStatic(nn.Module):
    def __init__(
        self,
        input_dim=5,
        output_dim=32,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.foundation = Foundation_local(
            input_dim=10, # B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
            depths=[3, 3, 4, 4, 5],  # 128, 64, 32, 16, 8
            dims=[32, 32, 64, 64, 128],
            img_size=128,
            latent_dim=1024,
            dropout=None,
            activation=nn.GELU(),
        )

        self.encoder = self.foundation.encoder # expects dim 32 input
        self.decoder = self.foundation.decoder

        self.stem = CNNBlock(self.input_dim, 32, activation=nn.GELU(), residual=False, chw=[self.input_dim, 128, 128])

    def forward(self, x):
        x = self.stem(x)
        embeddings, embeddings_cnn, skips, predictions = self.encoder(x)
        decoded = self.decoder(embeddings, skips)

        return decoded


class FloodPredictorHSL(nn.Module):
    def __init__(
        self,
        path_weights,
    ):
        super().__init__()
        
        self.foundation_spectral = FloodPredictorSpectral(input_dim=7, output_dim=32, path_weights=path_weights)
        self.foundation_static = FloodPredictorStatic(input_dim=15, output_dim=32)

        self.head = nn.Sequential(
            nn.Dropout2d(0.2),
            CNNBlock(64, 32, activation=nn.GELU(), residual=False, chw=[64, 128, 128]),
            CNNBlock(32, 2, activation=nn.GELU(), activation_out=nn.Softmax(dim=1), chw=[2, 128, 128])
        )

    def forward(self, x_spectral, x_static):
        x = torch.cat([
            self.foundation_spectral(x_spectral),
            self.foundation_static(x_static),
        ], dim=1)
        
        x = self.head(x)

        return x



def create_splits(dataset, train_size=0.8):
    # Calculate the size of each split
    train_len = int(len(dataset) * train_size)
    val_len = len(dataset) - train_len
    
    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    return train_dataset, val_dataset

def get_dataloaders(image_dir, label_dir, batch_size=16, train_size=0.8, num_workers=4, transform=None):
    # Initialize the full dataset
    full_dataset = Sentinel2Dataset(image_dir=image_dir, label_dir=label_dir, transform=transform)
    
    # Create training and validation splits
    train_dataset, val_dataset = create_splits(full_dataset, train_size=train_size)
    
    # Create DataLoaders for both splits
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader



def compute_stats(dataset,num_bands=7):
    mean = np.zeros(num_bands)
    std = np.zeros(num_bands)
    min_val = np.inf * np.ones(num_bands)
    max_val = -np.inf * np.ones(num_bands)
    nb_samples = 0
    
    for img,_ in dataset:
        nb_samples += 1
        for i in range(img.shape[0]):  # Assuming img.shape[0] is the number of channels
            channel_data = img[i].ravel()
            mean[i] += channel_data.mean()
            std[i] += channel_data.std()
            min_channel = channel_data.min()
            max_channel = channel_data.max()
            if min_channel < min_val[i]:
                min_val[i] = min_channel
            if max_channel > max_val[i]:
                max_val[i] = max_channel
    
    mean /= nb_samples
    std /= nb_samples
    
    return mean, std, min_val, max_val



def visualize_samples(dataloader, num_samples=3):
    fig, axs = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    for i_batch, sample_batched in enumerate(dataloader):
        images, labels = sample_batched
        for i in range(num_samples):
            img = images[i].numpy()[:,:,[3, 2, 1]]
            img = (img - img.min()) / (img.max() - img.min())*255
            label = labels[i].numpy()
            axs[i, 0].imshow(img[:, :, :3].astype("uint8")) # Show only RGB channels
            axs[i, 1].imshow(label, cmap='gray')
            axs[i, 0].set_title('Image')
            axs[i, 1].set_title('Label')
        break # Only show the first batch
    plt.show()


def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    since = time.time()

    val_acc_history = []
    val_f1_history = []
    train_f1_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf') 
    best_acc = 0.0
    best_f1_score = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs,_ = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    labels_class_indices = torch.argmax(labels, axis=1)
                    corrects = torch.sum(preds == labels_class_indices)
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels_class_indices.cpu().numpy())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += corrects
        
            
            all_preds = np.concatenate(all_preds).flatten()
            all_labels = np.concatenate(all_labels).flatten()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / (len(dataloaders[phase].dataset) * 128 * 128)
            
            epoch_f1_score = f1_score(all_labels, all_preds,average='weighted')

            if phase == 'train':
                train_f1_history.append(epoch_f1_score)
                print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_f1_score))
            # else:
            #     val_f1_history.append(epoch_f1_score)
            #     print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_f1_score))

                if epoch_f1_score > best_f1_score:
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_f1_score = epoch_f1_score
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best train F1: {:.4f}'.format(best_f1_score))

    model.load_state_dict(best_model_wts)
    
    print('Saving..')
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save({
        'model': best_model_wts,
        'loss': best_loss,
        'epoch': epoch,
        'acc': best_acc,
        'f1_score': best_f1_score,
    }, './checkpoint/ckpt_best_f1.pth')
    
    return model, val_f1_history        


def model_prediction(model, input_dir, device):

    model.eval()
    input = os.listdir(input_dir)
    for img in input:
        img_name = os.path.join(input_dir, img)
        with rasterio.open(img_name) as img_file:
            img = img_file.read([1, 2, 3, 4, 5, 6, 7])  # Reading the first 7 bands
            img = np.clip(img/10000, 0, 1)  # Clip values to 0-10000 range and normalize to 0-1
        img = torch.from_numpy(img).float()
        img = F.interpolate(img.unsqueeze(0), size=(512, 512), mode='nearest').squeeze(0)
        img = img.to(device)
        with torch.no_grad():
            output = model(img)
            _, preds = torch.max(output, 1)
            preds = preds.cpu().numpy()
            plt.imshow(preds)
            plt.show()





def calculate_class_weights(trainloader, num_classes):
    # Initialize counters for all classes
    class_counts = torch.zeros(num_classes)

    # Iterate over the DataLoader
    for _, mask, in trainloader:  # Assuming the DataLoader returns (input, mask, other_data)
        for class_id in range(num_classes):
            class_counts[class_id] += (mask == class_id).sum()


    class_counts[class_counts == 0] = 1

    inverse_freq = 1.0 / np.array(class_counts)
    
    # Normalize weights to sum to the number of classes
    weights = inverse_freq / np.sum(inverse_freq) * len(class_counts)
    
    return weights


    # Calculate weights inversely proportional to class counts
    class_weights = class_counts.max() / class_counts


    return class_weights

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, input, target):
        axes = tuple(range(1, input.dim()))
        intersect = (input * target).sum(dim=axes)
        union = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)
        loss = 1 - (2 * intersect + self.smooth) / (union + self.smooth)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)
        loss = - (target * torch.pow((1 - input), self.gamma) * torch.log(input) +
                  (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))
        return loss.mean()


class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.dice_loss(input, target) + self.focal_loss(input, target)
        return loss


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def model_prediction(model, input_dir, device):

    model.eval()
    input = os.listdir(input_dir)
    for img in input:
        img_name = os.path.join(input_dir, img)
        with rasterio.open(img_name) as img_file:
            img = img_file.read([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # Reading all 12 bands
            min = np.array([-1.393e+03, -1.169e+03, -7.220e+02, -6.840e+02, -4.120e+02, -3.350e+02, -2.580e+02, 6.400e+01, -9.999e+03, 8.000e+00, 1.000e+01, 0.000e+00]).reshape(-1, 1, 1)
            max = np.array([ 6568,  9659, 11368, 12041, 15841, 15252, 14647, 255, 4245, 4287, 100, 111]).reshape(-1, 1, 1)

            hot_encoder = SpatialHOT(
            classes=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
            device="cuda",
            )
            # img = np.clip(img/10000, 0, 1)  # Clip values to 0-10000 range and normalize to 0-1
            # Normalize the data with min-max scaling to 0-1 range
            img = img_file.read([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # Reading all 12 bands
            img_spectral = img[0:7]
            img_spectral = img_spectral/10000  # Clip values to 0-10000 range and normalize to 0-1
            img_static = img[[7,8,9,11],:,:]

            img_static = (img_static - min[[7,8,9,11],:,:]) / (max[[7,8,9,11],:,:] - min[[7,8,9,11],:,:])
            img_static_lc = hot_encoder(img[10]).cpu().numpy()


        img = np.concatenate([img_spectral, img_static, img_static_lc], axis=0)
        img = torch.from_numpy(img).float()


        # Upsample image and label to 512x512
        img = F.interpolate(img.unsqueeze(0), size=(512, 512), mode='bilinear')
        img = img.to(device)
        with torch.no_grad():
            output = model(img)
            _, preds = torch.max(output[0], 1)
            preds = preds.cpu().numpy().squeeze()
            preds = preds.astype(np.uint8)
            saved_dir = '/home/zhangz65/NASA_model/satlas/Adamw_focal_1e-4_f1_12bands_100_0_aug_affine_data_pre_new/pred/'
            Image.fromarray(preds).save(saved_dir + img_name.split('/')[-1].replace('.tif', '.png'))
            


class VotingClassifier:
    def __init__(self, models):
        self.models = models

    
    def predict(self, input_dir):
        total_predictions = []

        input = os.listdir(input_dir)
        for img in input:
            img_name = os.path.join(input_dir, img)
            with rasterio.open(img_name) as img_file:
                img = img_file.read([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # Reading all 12 bands
                min = np.array([-1.393e+03, -1.169e+03, -7.220e+02, -6.840e+02, -4.120e+02, -3.350e+02, -2.580e+02, 6.400e+01, -9.999e+03, 8.000e+00, 1.000e+01, 0.000e+00]).reshape(-1, 1, 1)
                max = np.array([ 6568,  9659, 11368, 12041, 15841, 15252, 14647, 255, 4245, 4287, 100, 111]).reshape(-1, 1, 1)

                hot_encoder = SpatialHOT(
                classes=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
                device="cuda",
                )
                # img = np.clip(img/10000, 0, 1)  # Clip values to 0-10000 range and normalize to 0-1
                # Normalize the data with min-max scaling to 0-1 range
                img = img_file.read([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # Reading all 12 bands
                img_spectral = img[0:7]
                img_spectral = img_spectral/10000  # Clip values to 0-10000 range and normalize to 0-1
                img_static = img[[7,8,9,11],:,:]

                img_static = (img_static - min[[7,8,9,11],:,:]) / (max[[7,8,9,11],:,:] - min[[7,8,9,11],:,:])
                img_static_lc = hot_encoder(img[10]).cpu().numpy()


            img = np.concatenate([img_spectral, img_static, img_static_lc], axis=0)
            img = torch.from_numpy(img).float().unsqueeze(0)
            img = img.to(device)


            # Aggregate predictions from all models
            all_preds = []
            for model in self.models:
                if model == model_phieo:
                    with torch.no_grad():
                        outputs = model(img[:, :7], img[:, 7:])
                        _, predicted_class = torch.max(outputs, 1)
                elif model == model_satlas_Ladsat:
                    img_satlas_Landsat = F.interpolate(img, size=(512, 512), mode='bilinear')
                
                    with torch.no_grad():
                        outputs= model(img_satlas_Landsat)
                        _, predicted_class = torch.max(outputs[0], 1)
                elif img_satlas_Landsat is not None:
                    with torch.no_grad():
                        outputs= model(img_satlas_Landsat)
                        _, predicted_class = torch.max(outputs[0], 1)
                        

                all_preds.append(predicted_class.cpu().numpy())
            
            # Perform majority voting
            all_preds = np.array(all_preds)
            from scipy import stats
            voted_preds, _ = stats.mode(all_preds, axis=0)
            voted_preds = voted_preds[0]

            preds = voted_preds.astype(np.uint8)
            saved_dir = '/home/zhangz65/NASA_model/satlas/voting/pred/'
            Image.fromarray(preds).save(saved_dir + img_name.split('/')[-1].replace('.tif', '.png'))
            



        
        return preds

    def evaluate(self, data_loader):
        correct = 0
        total = 0
        predictions = self.predict(data_loader)
        for idx, (data, target) in enumerate(data_loader):
            total += target.size(0)
            correct += (predictions[idx] == target.numpy()).sum()
        
        accuracy = 100 * correct / total
        return accuracy
    
def load_model(model, model_path):
    state = torch.load(model_path)
    model.load_state_dict(state['model'])
    return model


if __name__ == "__main__":
    

    set_seed(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    multiprocessing.set_start_method('spawn', True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




    # image_dir = '/home/zhangz65/NASA_model/satlas/Track2/train/images'
    # label_dir = '/home/zhangz65/NASA_model/satlas/Track2/train/labels'
    image_dir = '/mmfs1/scratch/hpc/00/zhangz65/satlas/data/Track2/train/images'
    label_dir = '/mmfs1/scratch/hpc/00/zhangz65/satlas/data/Track2/train/labels'
    # Experiment Arguments
    batch_size = 4
    num_workers = 4
    max_epochs = 400
    fast_dev_run = False
    val_step = 1 
    # weights = torch.tensor([1.0000, 3.0205]).to(device)  # Assuming you have 2 classes
    weights = torch.tensor([0.51330465, 1.4866954]).to(device)  # Assuming you have 2 classes 
    criterion = FocalLoss()
    # criterion = nn.CrossEntropyLoss(weight=weights)  
    # mean = [396.46749898, 494.62101716, 822.31995507,973.67493873, 2090.1119281, 1964.05106209, 1351.27389706]
    # std = [145.5476537, 182.14385468,236.99994894, 315.72007761,692.93872549, 746.6220384, 572.43704044]
    # Assuming your image directory is correctly set
    
    # dataset = Sentinel2Dataset(image_dir, label_dir, transform=None)
    # num_bands = 12
    # mean, std, min, max = compute_stats(dataset,num_bands = num_bands)
    # print(f"Mean: {mean}, Std: {std}, Min: {min}, Max: {max}")

    augment = v2.Compose(
        [
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([
                v2.RandomAffine((0, 360), translate=(0.1, 0.1)),
            ], p=0.5)
        ]
    )

    # Get DataLoaders
    # train_loader, val_loader = get_dataloaders(image_dir, label_dir, batch_size=batch_size, train_size=1.0, transform=augment,num_workers=num_workers)
    # dataloaders_dict = {'train': train_loader, 'val': val_loader}

        # Example usage
    # num_classes = 2
    # ignored_classes = None  # Classes you don't want to train on
    # weights = calculate_class_weights(dataloaders_dict['train'], num_classes)
    # print(weights)


    

                          
    # Visualize some training samples
    # visualize_samples(train_loader, num_samples=3)


    print("DataLoaders created successfully!")

    # LandSat satlas Model

    model_satlas_Ladsat = weights_manager.get_pretrained_model("Landsat_SwinB_SI", fpn=True, head=satlaspretrain_models.Head.SEGMENT, num_categories=2)
        

        # Replace first layer's input channels with the task's number input channels.
    first_layer =model_satlas_Ladsat.backbone.backbone.features[0][0]
    model_satlas_Ladsat.backbone.backbone.features[0][0] = torch.nn.Conv2d(22,
                        first_layer.out_channels,
                        kernel_size=first_layer.kernel_size,
                        stride=first_layer.stride,
                        padding=first_layer.padding,
                        bias=(first_layer.bias is not None))
    
    model_satlas_Ladsat = load_model(model_satlas_Ladsat, '/home/zhangz65/NASA_model/satlas/Adamw_focal_1e-4_f1_12bands_100_0_aug_affine_data_pre_new/checkpoint/ckpt_best_f1.pth')

    print(model_satlas_Ladsat)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_satlas_Ladsat = model_satlas_Ladsat.to(device)

    # Sentinel2 satlas Model

    model_satlas_Sentinel = weights_manager.get_pretrained_model("Sentinel2_SwinB_SI_MS", fpn=True, head=satlaspretrain_models.Head.SEGMENT, num_categories=2)
        

        # Replace first layer's input channels with the task's number input channels.
    first_layer =model_satlas_Sentinel.backbone.backbone.features[0][0]
    model_satlas_Sentinel.backbone.backbone.features[0][0] = torch.nn.Conv2d(22,
                        first_layer.out_channels,
                        kernel_size=first_layer.kernel_size,
                        stride=first_layer.stride,
                        padding=first_layer.padding,
                        bias=(first_layer.bias is not None))

    print(model_satlas_Sentinel)
    model_satlas_Sentinel = load_model(model_satlas_Sentinel, '/home/zhangz65/NASA_model/satlas/Adamw_focal_1e-4_f1_12bands_100_0_aug_affine_data_pre_new_sentinel2/checkpoint/ckpt_best_f1.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_satlas_Sentinel = model_satlas_Sentinel.to(device)

    #Phieo model

    model_phieo = FloodPredictorHSL(path_weights="/home/zhangz65/NASA_model/satlas/voting/Adamw_f1_90_10_12_bands_feature_fusion_e25/phileo-precursor_v08_e025.pt")
    # model = FloodPredictorHSL(input_dim=7, output_dim=2, path_weights="phileo-precursor_v01_e027.pt")
    model_phieo = load_model(model_phieo, '/home/zhangz65/NASA_model/satlas/voting/Adamw_f1_90_10_12_bands_feature_fusion_e25/checkpoint/ckpt_best_f1.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_phieo = model_phieo.to(device)

    model_satlas_Ladsat.eval()
    model_satlas_Sentinel.eval()
    model_phieo.eval()

    ensemble = VotingClassifier(models=[model_satlas_Ladsat, model_satlas_Sentinel, model_phieo])

    predictions = ensemble.predict('/home/zhangz65/NASA_model/satlas/Test Set Track 2/images')

    # testing accuracy