import dataset_module
import load_data
import pathnet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch.optim as optim

# 一些常量
learning_rate = 0.1
batch_size = 64
epochs = 1 # 将整体数据迭代多少代


if __name__ == '__main__':
    # 先制作 DataLoader
    trainset = dataset_module.PN_dataset('train_pathnet',
                                         transform=transforms.ToTensor())
    train_loader = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2)


    # 定义 网络
    net = pathnet.PathNet()

    # 定义学习相关的一系列参数 , 优化器 等
    loss = nn.MSELoss() # 损失函数 (1/n)*|x-y|^2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5, gamma=0.05) # 每5个epoch 调整一下学习率

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1,epochs))
        print('--'*10)
        net.train()
        lossss = 0 # 用来计算loss
        for batch_id, sample in enumerate(train_loader):
            input_batch = sample['input']
            labels = sample['label']
            outputs = net(input_batch)
            losses = loss(outputs, labels)
            losses.backward()
            optimizer.zero_grad()
            optimizer.step()
            lossss += losses.item()
        print('平均loss = {}'.format(lossss / batch_size))
        scheduler.step()



