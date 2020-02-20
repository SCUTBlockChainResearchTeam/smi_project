import torch.optim as optim
import func_train_and_valid as ftav
import dataset_module
from torch.utils.data import DataLoader
from torchvision import transforms
import net_modules
import torch.nn as nn
"""
@Constant:
learning_rate = 0.1
batch_size = 64
epoch = 5 # 测试阶段 真正训练需要百代以上
model_save_path = ''
"""

if __name__ == '__main__':
    '''
    overlap net 的调用示例
    '''
    # 构建数据集
    trainset = dataset_module.ON_dataset(transform=transforms.ToTensor())
    validset = dataset_module.ON_dataset(mode='valid_pathnet', transform=transforms.ToTensor())
    # 做一个加载器 用来按batch给网络喂数据
    trainloader = DataLoader(trainset, batch_size=ftav.CONST_BATCH_SIZE, shuffle=True, num_workers=4)
    validloader = DataLoader(validset, batch_size=ftav.CONST_BATCH_SIZE, shuffle=True, num_workers=4)
    loader_dict = {'train': trainloader, 'valid': validloader}  # 做一个字典  用来切换

    # 初始化网络
    net = net_modules.OverlapNet()
    # 初始化 loss计算标准
    criterion = nn.MSELoss()
    # 初始化一个Adam优化器 带动量
    optimizer = optim.Adam(net.parameters(), lr=ftav.CONST_LR)
    # 初始化一个学习率衰减器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.05)  # 每5个epoch 调整一下学习率

    model, val_acc_history, val_losses, train_acc_history, train_losses = ftav.func_train_overlap(model=net,
                                                                                          dataloaders=loader_dict,
                                                                                          criterion=criterion,
                                                                                          optimizer=optimizer,
                                                                                          scheduler=scheduler,
                                                                                          model_save_path='test.pth',
                                                                                          num_epoch=20)
    print('TEST')


