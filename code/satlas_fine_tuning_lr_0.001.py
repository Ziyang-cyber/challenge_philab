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

weights_manager = satlaspretrain_models.Weights()

print(torch.__version__)

print('test')

class Sentinel2Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.mean = np.array([396.46749898, 494.62101716, 822.31995507,973.67493873, 2090.1119281, 1964.05106209, 1351.27389706])
        self.std = np.array([145.5476537, 182.14385468,236.99994894, 315.72007761,692.93872549, 746.6220384, 572.43704044])
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        label_name = os.path.join(self.label_dir, self.images[idx].replace('.tif', '.png'))

        # Read Sentinel-2 TIFF image (first 7 bands)
        with rasterio.open(img_name) as img_file:
            img = img_file.read([1, 2, 3, 4, 5, 6, 7])  # Reading the first 7 bands
            img = np.clip(img/10000, 0, 1)  # Clip values to 0-10000 range and normalize to 0-1

            


        # Read label PNG image
        label = np.array(Image.open(label_name))

        # counts = np.bincount(label.flatten())

        # # Assuming class labels are 0 and 1
        # num_class_0 = counts[0]
        # num_class_1 = counts[1]

        # print(f"Number of Class 0 pixels: {num_class_0}")
        # print(f"Number of Class 1 pixels: {num_class_1}")

        # plt.imshow(label)

        # print(f"Image shape: {img.shape}")

        # Convert label to binary format if necessary
        # label = (label > 0).astype(np.float32)

        # Convert arrays to PyTorch tensors
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).float()  # Add channel dimension to label

        

        # mean = torch.from_numpy(self.mean).float().view(7, 1, 1)  # Reshape mean for broadcasting
        # std = torch.from_numpy(self.std).float().view(7, 1, 1)  # Reshape std for broadcasting
        # img = (img - mean) / std  # Apply standardization
        # print(img.max(), img.min())

        # Apply transformations if any
        if self.transform:
            augmented = self.transform(image=img, mask=label)
            img = augmented['image']
            label = augmented['mask']

        # Upsample image and label to 512x512
        img = F.interpolate(img.unsqueeze(0), size=(512, 512), mode='nearest').squeeze(0)
        # label = F.interpolate(label.unsqueeze(0), size=(512, 512), mode='nearest').squeeze(0).squeeze(0)  # Remove added channel dimension for label

        # Convert label back to long tensor (for categorical data, if necessary)
        # num_classes = 2
        label = label.long()
        # label = F.one_hot(label, num_classes)  # Convert to one-hot encoding
        # label = label.permute(2, 0, 1).float() # Reshape to (C, H, W) format

        return img, label
    


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



def compute_stats(dataset):
    mean = np.zeros(7)
    std = np.zeros(7)
    min_val = np.inf * np.ones(7)
    max_val = -np.inf * np.ones(7)
    nb_samples = 0
    
    for img in dataset:
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

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf') 

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

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
                    # labels_class_indices = torch.argmax(labels, axis=1)
                    corrects = torch.sum(preds == labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += corrects

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / (len(dataloaders[phase].dataset) * inputs.size(2) * inputs.size(3))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    print('Saving..')
    state = {
        'model': best_model_wts,
        'loss': best_loss,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth')
    
    return model, val_acc_history

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

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




    # image_dir = '/home/zhangz65/NASA_model/satlas/Track2/train/images'
    # label_dir = '/home/zhangz65/NASA_model/satlas/Track2/train/labels'
    image_dir = '/mmfs1/scratch/hpc/00/zhangz65/satlas/data/Track2/train/images'
    label_dir = '/mmfs1/scratch/hpc/00/zhangz65/satlas/data/Track2/train/labels'
    # Experiment Arguments
    batch_size = 16
    num_workers = 4
    max_epochs = 200
    fast_dev_run = False
    val_step = 1 
    # weights = torch.tensor([1.0000, 3.0205]).to(device)  # Assuming you have 2 classes
    weights = torch.tensor([0.51330465, 1.4866954]).to(device)  # Assuming you have 2 classes 
    criterion = nn.CrossEntropyLoss(weight=weights)  # Assuming you have 2 classes
    # mean = [396.46749898, 494.62101716, 822.31995507,973.67493873, 2090.1119281, 1964.05106209, 1351.27389706]
    # std = [145.5476537, 182.14385468,236.99994894, 315.72007761,692.93872549, 746.6220384, 572.43704044]
    # Assuming your image directory is correctly set
    
    dataset = Sentinel2Dataset(image_dir, label_dir, transform=None)

    # mean, std, min, max = compute_stats(dataset)
    # print(f"Mean: {mean}, Std: {std}, Min: {min}, Max: {max}")

    # Get DataLoaders
    train_loader, val_loader = get_dataloaders(image_dir, label_dir, batch_size=batch_size, train_size=0.8, transform=None)
    dataloaders_dict = {'train': train_loader, 'val': val_loader}

        # Example usage
    # num_classes = 2
    # ignored_classes = None  # Classes you don't want to train on
    # weights = calculate_class_weights(dataloaders_dict['train'], num_classes)
    # print(weights)


    

                          
    # Visualize some training samples
    # visualize_samples(train_loader, num_samples=3)


    print("DataLoaders created successfully!")


    model = weights_manager.get_pretrained_model("Landsat_SwinB_SI", fpn=True, head=satlaspretrain_models.Head.SEGMENT, num_categories=2)
        

        # Replace first layer's input channels with the task's number input channels.
    first_layer =model.backbone.backbone.features[0][0]
    model.backbone.backbone.features[0][0] = torch.nn.Conv2d(7,
                        first_layer.out_channels,
                        kernel_size=first_layer.kernel_size,
                        stride=first_layer.stride,
                        padding=first_layer.padding,
                        bias=(first_layer.bias is not None))

    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer and learning rate scheduler
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop.
    # for epoch in range(max_epochs):
    #     print("Starting Epoch...", epoch)

    #     for data, target in train_loader:
    #         data = data.to(device)
    #         target = target.to(device)

    #         output, _ = model(data, target)

    #         loss = criterion(output, target)

            

    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #     print("Train Loss = ", loss)

    #     # Validation.
    #     if epoch % val_step == 0:
    #         model.eval()
    #         total_accuracy = 0
    #         with torch.no_grad():
    #             for val_data, val_target in val_loader:
    #                 val_data, val_target = val_data.to(device), val_target.to(device)
    #                 val_output = model(val_data)
    #                 val_accuracy = (val_output.argmax(dim=1) == val_target).float().mean().item()
    #                 total_accuracy += val_accuracy
            
    #         average_val_accuracy = total_accuracy / len(val_loader)
    #         print(f"Epoch {epoch}: Validation accuracy = {average_val_accuracy}")

    #         # Save the model if it has a better accuracy
    #         if average_val_accuracy > best_val_accuracy:
    #             best_val_accuracy = average_val_accuracy
    #             best_model_path = "/path/to/save/best_model.pt"
    #             torch.save(model.state_dict(), best_model_path)
    #             print(f"New best model saved with accuracy: {best_val_accuracy}")



    # Initialize the training code.
    # accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    # default_root_dir = os.path.join(tempfile.gettempdir(), "experiments")

    # trainer = Trainer(
    #     accelerator=accelerator,
    #     default_root_dir=default_root_dir,
    #     fast_dev_run=fast_dev_run,
    #     log_every_n_steps=1,
    #     min_epochs=1,
    #     max_epochs=max_epochs,
    # )

    # Train the model
    # trainer.fit(model, train_loader, val_loader)

    # Save the model

    model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=max_epochs)







# tensor = torch.zeros((1, 6, 128, 128), dtype=torch.float32)

# model = SatlasSegmentationTask()


# model.eval()
# output = model(tensor)
# print(output)


