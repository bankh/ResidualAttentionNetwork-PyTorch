from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
import os
import cv2
import time
import logging

from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel
#  from model.residual_attention_network_pre import ResidualAttentionModel
# # based https://github.com/liudaizong/Residual-Attention-Network

def generate_folder_name(model_arch, optimizer):
    return f'saved/{model_arch}_{optimizer}'

# Initialize logging
folder_name = generate_folder_name('ResidualAttentionModel', 'sgd')
os.makedirs(folder_name, exist_ok=True)
log_file = f'{folder_name}/training.log'
logging.basicConfig(filename=log_file, level=logging.INFO)

acc_best = 0
early_stop_counter = 0
early_stop_limit = 20
total_epoch = 300

def test(model, test_loader, btrain=False, model_file='model_92.pkl'):
    if not btrain:
        model.load_state_dict(torch.load(model_file))
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
    return correct / total

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((32, 32), padding=4),
    transforms.ToTensor()
])
test_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data/', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data/', train=False, transform=test_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=20, shuffle=False)

model = ResidualAttentionModel().cuda()
lr = 0.1
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)

for epoch in range(total_epoch):
    model.train()
    tims = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            log_message = f"Epoch [{epoch+1}/{total_epoch}], Iter [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}"
            print(log_message)
            logging.info(log_message)
    log_message = f'The epoch takes time: {time.time() - tims}'
    print(log_message)
    logging.info(log_message)
    print('Evaluate test set:')
    acc = test(model, test_loader, btrain=True)
    if acc > acc_best:
        acc_best = acc
        early_stop_counter = 0
        model_file = f'{folder_name}/model_92_sgd_ep{epoch+1}_acc{acc_best:.4f}.pkl'
        torch.save(model.state_dict(), model_file)
    else:
        early_stop_counter += 1
    if early_stop_counter >= early_stop_limit:
        break
