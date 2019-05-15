import os
import sys
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.ranking import roc_auc_score
from sklearn.model_selection import train_test_split
from PIL import Image
import multiprocessing

assert torch.cuda.is_available()

torch.backends.cudnn.benchmark=True

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("PyTorch: ", torch.__version__)
print("Numpy: ", np.__version__)

CPU_COUNT = multiprocessing.cpu_count()
print("CPUs: ", CPU_COUNT)

# Globals
# With small batch may be faster on P100 to do one 1 GPU
MULTI_GPU = True
CLASSES = 14
WIDTH = 224
HEIGHT = 224
CHANNELS = 3
LR = 0.0001
EPOCHS = 2 #100
# Can scale to max for inference but for training LR will be affected
# Prob better to increase this though on P100 since LR is not too low
# Easier to see when plotted
BATCHSIZE = 16 #64*2
IMAGENET_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGENET_RGB_SD = [0.229, 0.224, 0.225]

import sys, os
paths_to_append = [os.path.join(os.getcwd(), os.path.join(*(['Code',  'src'])))]
def add_path_to_sys_path(path_to_append):
    if not (any(path_to_append in paths for paths in sys.path)):
        sys.path.append(path_to_append)
[add_path_to_sys_path(crt_path) for crt_path in paths_to_append]

import azure_chestxray_utils

path= r'C:\Users\YanHaotian\workspace\AzureChestXRay_AMLWB\Code\azure-share'
amlWBSharedDir = path

prj_consts = azure_chestxray_utils.chestxray_consts()

data_base_input_dir=os.path.join(amlWBSharedDir,
                                 os.path.join(*(prj_consts.BASE_INPUT_DIR_list)))
data_base_output_dir=os.path.join(amlWBSharedDir,
                                  os.path.join(*(prj_consts.BASE_OUTPUT_DIR_list)))
nih_chest_xray_data_dir=os.path.join(data_base_input_dir,
                                     os.path.join(*(prj_consts.ChestXray_IMAGES_DIR_list)))
other_data_dir=os.path.join(data_base_input_dir,
                            os.path.join(*(prj_consts.ChestXray_OTHER_DATA_DIR_list)))
label_file = os.path.join(other_data_dir,'Data_Entry_2017.csv')

data_partitions_dir=os.path.join(data_base_output_dir,
                                os.path.join(*(prj_consts.DATA_PARTITIONS_DIR_list)))

import pickle
patient_id_partition_file = os.path.join(data_partitions_dir, 'train_test_valid_data_partitions.pickle')

with open(patient_id_partition_file, 'rb') as f:
    [train_set,valid_set,test_set, nih_annotated_set]=pickle.load(f)

print("train:{} valid:{} test:{} nih-annotated:{}".format(len(train_set), len(valid_set), \
                                                     len(test_set), len(nih_annotated_set)))


class XrayData(Dataset):
    def __init__(self, img_dir, lbl_file, patient_ids, transform=None):
        # Read labels-csv
        df = pd.read_csv(lbl_file)
        # Filter by patient-ids
        df = df[df['Patient ID'].isin(patient_ids)]
        # Split labels
        df_label = df['Finding Labels'].str.split(
            '|', expand=False).str.join(sep='*').str.get_dummies(sep='*')
        df_label.drop(['No Finding'], axis=1, inplace=True)

        # List of images (full-path)
        self.img_locs = df['Image Index'].map(lambda im: os.path.join(img_dir, im)).values
        # One-hot encoded labels (float32 for BCE loss)
        self.labels = df_label.values
        # Processing
        self.transform = transform

        print("Loaded {} labels and {} images".format(len(self.labels),
                                                      len(self.img_locs)))

    def __getitem__(self, idx):
        im_file = self.img_locs[idx]
        im_rgb = Image.open(im_file).convert('RGB')
        label = self.labels[idx]
        if self.transform is not None:
            im_rgb = self.transform(im_rgb)
        return im_rgb, torch.FloatTensor(label)

    def __len__(self):
        return len(self.img_locs)


def no_augmentation_dataset(img_dir, lbl_file, patient_ids, normalize):
    dataset = XrayData(img_dir, lbl_file, patient_ids,
                       transform=transforms.Compose([
                           transforms.Resize(WIDTH),
                           transforms.ToTensor(),
                           normalize]))
    return dataset

# Dataset for training
# Normalise by imagenet mean/sd
normalize = transforms.Normalize(IMAGENET_RGB_MEAN, IMAGENET_RGB_SD)
# todo
# Go wild here with the transforms
# https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
#__all__ = ["Compose", "ToTensor", "ToPILImage", "Normalize", "Resize", "Scale", "CenterCrop", "Pad",
#           "Lambda", "RandomCrop", "RandomHorizontalFlip", "RandomVerticalFlip", "RandomResizedCrop",
#           "RandomSizedCrop", "FiveCrop", "TenCrop", "LinearTransformation", "ColorJitter", "RandomRotation",
#           "Grayscale", "RandomGrayscale"]
train_dataset = XrayData(img_dir=nih_chest_xray_data_dir,
                         lbl_file=label_file,
                         patient_ids=train_set,
                         transform=transforms.Compose([
                             transforms.Resize(264),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomResizedCrop(size=WIDTH),
                             transforms.ColorJitter(0.15, 0.15),
                             transforms.RandomRotation(15),
                             transforms.ToTensor(),  # need to convert image to tensor!
                             normalize]))

valid_dataset = no_augmentation_dataset(nih_chest_xray_data_dir, label_file, valid_set, normalize)
test_dataset = no_augmentation_dataset(nih_chest_xray_data_dir, label_file, test_set, normalize)

def get_symbol(out_features=CLASSES, multi_gpu=MULTI_GPU):
    model = models.densenet.densenet121(pretrained=True)
    # Replace classifier (FC-1000) with (FC-14)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, out_features),
        nn.Sigmoid())
    if multi_gpu:
        model = nn.DataParallel(model)
    # CUDA
    model.cuda()
    return model

def init_symbol(sym, lr=LR):
    # torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    opt = optim.Adam(sym.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.BCELoss()
    scheduler = ReduceLROnPlateau(opt, factor = 0.1, patience = 5, mode = 'min')
    return opt, criterion, scheduler

def compute_roc_auc(data_gt, data_pd, mean=True, classes=CLASSES):
    roc_auc = []
    data_gt = data_gt.cpu().numpy()
    data_pd = data_pd.cpu().numpy()
    for i in range(classes):
        roc_auc.append(roc_auc_score(data_gt[:, i], data_pd[:, i]))
    if mean:
        roc_auc = np.mean(roc_auc)
    return roc_auc

def train_epoch(model, dataloader, optimizer, criterion, epoch, batch=BATCHSIZE):
    model.train()
    print("Training epoch {}".format(epoch+1))
    loss_val = 0
    loss_cnt = 0
    for data, target in dataloader:
        # Get samples
        data = Variable(torch.FloatTensor(data).cuda())
        target = Variable(torch.FloatTensor(target).cuda())
        # Init
        optimizer.zero_grad()
        # Forwards
        output = model(data)
        # Loss
        loss = criterion(output, target)
        # Back-prop
        loss.backward()
        optimizer.step()
         # Log the loss
        loss_val += loss.data[0]
        loss_cnt += 1
    print("Training loss: {0:.4f}".format(loss_val/loss_cnt))

def valid_epoch(model, dataloader, criterion, epoch, phase='valid', batch=BATCHSIZE):
    model.eval()
    if phase == 'testing':
        print("Testing epoch {}".format(epoch+1))
    else:
        print("Validating epoch {}".format(epoch+1))
    out_pred = torch.FloatTensor().cuda()
    out_gt = torch.FloatTensor().cuda()
    loss_val = 0
    loss_cnt = 0
    for data, target in dataloader:
        # Get samples
        data = Variable(torch.FloatTensor(data).cuda(), volatile=True)
        target = Variable(torch.FloatTensor(target).cuda(), volatile=True)
         # Forwards
        output = model(data)
        # Loss
        loss = criterion(output, target)
        # Log the loss
        loss_val += loss.data[0]
        loss_cnt += 1
        # Log for AUC
        out_pred = torch.cat((out_pred, output.data), 0)
        out_gt = torch.cat((out_gt, target.data), 0)
    loss_mean = loss_val/loss_cnt
    if phase == 'testing':
        print("Test-Dataset loss: {0:.4f}".format(loss_mean))
        print("Test-Dataset AUC: {0:.4f}".format(compute_roc_auc(out_gt, out_pred)))

    else:
        print("Validation loss: {0:.4f}".format(loss_mean))
        print("Validation AUC: {0:.4f}".format(compute_roc_auc(out_gt, out_pred)))
    return loss_mean

def print_learning_rate(opt):
    for param_group in opt.param_groups:
        print("Learining rate: ", param_group['lr'])

# DataLoaders
# num_workers=4*CPU_COUNT
# pin_memory=True
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCHSIZE,
                          shuffle=True, num_workers=0, pin_memory=False)

valid_loader = DataLoader(dataset=valid_dataset, batch_size=8*BATCHSIZE,
                          shuffle=False, num_workers=0, pin_memory=False)

test_loader = DataLoader(dataset=test_dataset, batch_size=8*BATCHSIZE,
                         shuffle=False, num_workers=0, pin_memory=False)


# Load symbol
azure_chest_xray_sym = get_symbol()

optimizer, criterion, scheduler = init_symbol(azure_chest_xray_sym)

# Original CheXNet ROC AUC = 0.841
loss_min = float("inf")
stime = time.time()

# No-training
valid_epoch(azure_chest_xray_sym, valid_loader, criterion, -1)

# Main train/val/test loop
for j in range(EPOCHS):
    train_epoch(azure_chest_xray_sym, train_loader, optimizer, criterion, j)
    loss_val = valid_epoch(azure_chest_xray_sym, valid_loader, criterion, j)
    test_loss_val = valid_epoch(azure_chest_xray_sym, test_loader, criterion, j, 'testing')
    # LR Schedule
    scheduler.step(loss_val)
    print_learning_rate(optimizer)
    # todo: tensorboard hooks
    # Logging
    if loss_val < loss_min:
        print("Loss decreased. Saving ...")
        loss_min = loss_val
        torch.save({'epoch': j + 1,
                    'state_dict': azure_chest_xray_sym.state_dict(),
                    'best_loss': loss_min,
                    'optimizer' : optimizer.state_dict()}, 'best_azure_chest_xray_model_v2.pth.tar')
    etime = time.time()
    print("Epoch time: {0:.0f} seconds".format(etime-stime))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")