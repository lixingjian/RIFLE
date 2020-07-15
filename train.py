# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torchvision.models import resnet50, inception_v3, densenet169# mobilenet_v2
from efficientnet_pytorch import EfficientNet
import time
import argparse
import math
import pickle
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description = 'RIFLE')
parser.add_argument('--data_dir')
parser.add_argument('--save_model', default = '')
parser.add_argument('--base_model', default = 'resnet50')
parser.add_argument('--epochs', type = int, default = 40)
parser.add_argument('--image_size', type = int, default = 224)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--lr_init', type = float, default = 0.01)
parser.add_argument('--reg_type', choices = ['l2', 'l2_sp'], default = 'l2')
parser.add_argument('--alpha', type = float, default = 0.01)
parser.add_argument('--beta', type = float, default = 0.0001)

parser.add_argument('--fc_reinit_times', type = int, default = 0)
parser.add_argument('--fc_lr_util', choices = ['none', 'cyclic'], default = 'none')
parser.add_argument('--fc_reinit', type = int, default = 0)
args = parser.parse_args()

print(torch.__version__)
print(args)
device = torch.device("cuda:0")

num_periods = args.fc_reinit_times + 1

image_size = args.image_size
crop_size = {299: 320, 224: 256, 300: 320}
resize = crop_size[image_size]
hflip = transforms.RandomHorizontalFlip()
rcrop = transforms.RandomCrop((image_size, image_size))
ccrop = transforms.CenterCrop((image_size, image_size))
totensor = transforms.ToTensor()
cnorm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #mean and std for imagenet

def transform_compose_train():
    r = [transforms.Resize(resize), hflip, rcrop, totensor, cnorm]
    return transforms.Compose(r)

def transform_compose_test():
    r = [transforms.Resize(resize), ccrop, totensor, cnorm]
    return transforms.Compose(r)

data_transforms = {'train': transform_compose_train(), 'test': transform_compose_test()}
set_names = list(data_transforms.keys())
image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x),
                                          data_transforms[x])
                  for x in set_names}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=4)
              for x in set_names}
dataset_sizes = {x: len(image_datasets[x]) for x in set_names}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

def get_base_model(base_model):
    if base_model.startswith('efficientnet'):
        model = EfficientNet.from_pretrained(base_model)
    else:
        model = eval(base_model)(pretrained = True)
    if base_model == 'inception_v3':
        model.aux_logits = False
    return model

class DropConnect(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = 1.0 - ratio

    def forward(self, x):
        if not self.training:
            return x
        random_tensor = self.ratio
        random_tensor += torch.rand([x.shape[0], 1], dtype=torch.float, device=x.device)
        random_tensor.requires_grad_(False)
        return x / self.ratio * random_tensor.floor()

model_source = get_base_model(args.base_model)
model_source.to(device)
for param in model_source.parameters():
    param.requires_grad = False
model_source.eval()

model_source_weights = {}
for name, param in model_source.named_parameters():
    model_source_weights[name] = param.detach() 

model_target = get_base_model(args.base_model)

if args.base_model.startswith('densenet'):
    fc_layer = model_target.classifier
    num_features = fc_layer.weight.shape[1]
    model_target.classifier = nn.Linear(num_features, num_classes)
    fc_layer = model_target.classifier
elif args.base_model.startswith('mobilenet'):
    fc_layer = model_target.classifier
    num_features = 1280
    model_target.classifier = nn.Linear(num_features, num_classes)
    fc_layer = model_target.classifier
elif args.base_model.startswith('efficientnet'):
    fc_layer = model_target._fc
    num_features = fc_layer.weight.shape[1]
    model_target._fc = nn.Linear(num_features, num_classes)
    fc_layer = model_target._fc
else:
    fc_layer = model_target.fc
    num_features = 2048
    model_target.fc = nn.Linear(num_features, num_classes)
    fc_layer = model_target.fc
    fc_name = 'fc.'
    assert fc_layer is model_target.fc

model_target.to(device)

def reg_classifier(model):
    l2_cls = torch.tensor(0.).to(device)
    for name, param in model.named_parameters():
        if name.startswith(fc_name):
            l2_cls += 0.5 * torch.norm(param) ** 2
    return l2_cls

def reg_l2sp(model):
    fea_loss = torch.tensor(0.).to(device)
    for name, param in model.named_parameters():
        if not name.startswith(fc_name):
            fea_loss += 0.5 * torch.norm(param - model_source_weights[name]) ** 2
    return fea_loss

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    for epoch in range(num_epochs):
        lr_cnn = 0.5 * args.lr_init * (1 + math.cos(math.pi * epoch / num_epochs))
        if args.fc_lr_util == 'cyclic':
            u = num_epochs // num_periods
            lr_fc = 0.5 * args.lr_init * (1 + math.cos(math.pi * (epoch % u) / u))
        else:
            lr_fc = lr_cnn
        optimizer.param_groups[1]['lr'] = lr_cnn
        optimizer.param_groups[0]['lr'] = lr_fc
        print('Epoch {}/{}, LR {}/{}'.format(epoch, num_epochs - 1, lr_cnn, lr_fc))
        print('-' * 10)
        #confusion_matrix.reset()
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            nstep = len(dataloaders[phase])
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss_main = criterion(outputs, labels)
                    loss_classifier = 0
                    loss_feature = 0
                    if not args.reg_type == 'l2':
                        loss_classifier = reg_classifier(model) 
                    if args.reg_type == 'l2_sp':
                        loss_feature = reg_l2sp(model)
                    loss = loss_main + args.alpha * loss_feature + args.beta * loss_classifier
 
                    _, preds = torch.max(outputs, 1)
                    #confusion_matrix.add(preds.data, labels.data)
                    if phase == 'train' and  i % 10 == 0:
                        corr_sum = torch.sum(preds == labels.data)
                        step_acc = corr_sum.double() / len(labels)
                        print('step: %d/%d, loss = %.4f(%.4f, %.4f, %.4f), top1 = %.4f' %(i, nstep, loss, loss_main, args.alpha * loss_feature, args.beta * loss_classifier, step_acc))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                    
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} epoch: {:d} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch, epoch_loss, epoch_acc))
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            if epoch == num_epochs - 1:
                print('{} epoch: last Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
        print()
        if num_periods > 1 and (epoch + 1) % (num_epochs // num_periods) == 0:
            print('reinit fc layer')
            fc_layer.reset_parameters()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model

param_fc = list(map(id, fc_layer.parameters()))
param_cnn = filter(lambda p: id(p) not in param_fc, model_target.parameters())
if args.reg_type == 'l2':
    optimizer_ft = optim.SGD([{'params': fc_layer.parameters()}, {'params': param_cnn}],
                        lr=args.lr_init, momentum=0.9, weight_decay = 1e-4)
else:
    optimizer_ft = optim.SGD([{'params': fc_layer.parameters()}, {'params': param_cnn}],
                        lr=args.lr_init, momentum=0.9)
num_epochs = args.epochs 
num_epochs = num_periods * (num_epochs // num_periods)

criterion = nn.CrossEntropyLoss()
train_model(model_target, criterion, optimizer_ft, None, num_epochs)

