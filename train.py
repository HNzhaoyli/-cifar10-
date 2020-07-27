import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms

import time
import os
from model import Net
from lr_scheduler import LRScheduler
import matplotlib.pyplot as plt
from triplet_loss import TripletLoss,CrossEntropyLabelSmooth

os.environ['CUDA_VISIBALE_DEVICES'] = '0,1'


train_dataset = dsets.CIFAR10(root='/ml/pycifar',  # 选择数据的根目录
                            train=True,  # 选择训练集
                            transform=transforms.ToTensor(),  # 转换成tensor变量
                            download=True)  # 从网络上download图片
test_dataset = dsets.CIFAR10(root='/ml/pycifar',  # 选择数据的根目录
                           train=False,  # 选择测试集
                           transform=transforms.ToTensor(),  # 转换成tensor变量
                           download=True)  # 从网络上download图片
dataset_sizes = len(train_dataset)
# 加载数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=32,  # 使用批次数据
                                           shuffle=True)  # 将数据打乱
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=32,
                                          shuffle=True)
print(len(train_dataset))
def train_model(model,num_epoch):
    since = time.time()
    lr_list = []
    for epoch in range(num_epoch):
        print('Epoh {}/{}'.format(epoch, num_epoch - 1))
        print('-' * 20)
        running_loss = 0.0
        running_corrects = 0.0

        lr_scheduler = LRScheduler(base_lr=0.05, step=[50, 80], factor=0.1, warmup_epoch=10, warmup_begin_lr=3e-4)
        lr = lr_scheduler.update(epoch)
        lr_list.append(lr)
        print(lr)
       
        ignored_params = list(map(id, model.module.liner1.parameters()))
                                         # map(id,model.module.liner2.parameters())))
        ignored_params += (list(map(id,model.module.liner2.parameters())) )

        base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * lr},
            {'params': model.module.liner1.parameters(), 'lr': lr},
            {'params': model.module.liner2.parameters(), 'lr': lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        #optimizer = torch.optim.SGD(model.parameters(),lr)

        for data in train_loader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            now_batch_size, c, h, w = inputs.shape
            if now_batch_size < 32:
                continue
            inputs, labels = Variable(inputs), Variable(labels)

            criterion = nn.CrossEntropyLoss()
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out,labels)
            running_loss += loss
            _, preds = torch.max(out.data, 1)
            running_corrects += float(torch.sum(preds == labels.data))
            epoch_acc = running_corrects / dataset_sizes
            loss.backward()
            optimizer.step()
        print('Epoch:{}   Loss: {:.4f}  acc: {:.4f} '.format(epoch, running_loss,epoch_acc))
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    plt.plot(lr_list)
    plt.show()
    print('Finished training....')

if __name__ == '__main__':
    net = Net()
    net = nn.DataParallel(net)
    print(net)
    net = net.cuda()
    modnet = train_model(net,100)
