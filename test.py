# 孙浩博
# 2021/4/29 11:11
# Using the trained model to test

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import Resnet
from Resnet import ResNet18
import numpy as np
import matplotlib.pyplot as plt
import os

# 是否在GPU上面跑
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 模型路径
pt_path = './'
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root='./test',
                                    transform=test_transform,
                                    )

test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, )
# 选择网络
# item = input('select the training model\n1:resnet18       2:self_built resnet\n')
# resnet18 = Resnet.ResNet18()
# if item == '1':
#     resnet18 = models.resnet18()
#     resnet18.load_state_dict(torch.load("./resnet18/Palmprint_model_288.pt"))
# elif item == '2':
#     resnet18 = Resnet.ResNet18()
#     resnet18.load_state_dict(torch.load("./resnet/Palmprint_model_270.pt"))

resnet18 = Resnet.ResNet18()
resnet18.load_state_dict(torch.load(pt_path))

model = resnet18.to(device)

with torch.no_grad():
    model.eval()
    number = 0
    total = 0
    test_data_size = len(test_loader)
    for batch_idx, (x, label) in enumerate(test_loader):
        # 在gpu跑
        x = x.to(device)
        label = label.to(device)

        # 损失
        logits = model(x)

        # 计算acc
        pred = logits.argmax(dim=1)
        correct = torch.eq(pred, label).float().sum().item()
        number += correct
        total += x.size(0)
    avg_test_acc = number / total
    print('Test Accuracy of the model on the test images: {} %'.format(avg_test_acc))