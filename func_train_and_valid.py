import dataset_module
import copy
import load_data
import time
import net_modules
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
"""
@CONSTANT:  |learning_rate|batch_size|epoch

Here some API called to train and validate our path net and overlap net
These function try to return the training results to user 
you can change the configuration const var below to adjust the process of training

@tips: make sure you call .cpu() before you transfer the tensor in gpu to np.ndarray
"""
CONST_LR = 0.1  # our learning rate
CONST_BATCH_SIZE = 64 # how many samples included in a batch
CONST_EPOCH = 10 # this should be low if just run and take a look otherwise high means official training


# 这里我们要将训练和验证放在一起 原因是因为我们想从验证的效果中得到最好的模型并保存下来 filename参数指定模型保存路径
def func_train_path(model ,dataloaders, criterion, optimizer, scheduler, model_save_path = 'pathnet.pth', num_epoch= 10):
    # 记录一下时间
    start_time = time.time()

    # 初始准确率置0
    best_acc = 0

    # 检测一下当前设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 如果要从已有的存贮的模型中恢复之前训练的
    """
    # 这些是如果想要从之前已经保存的优良模型文件中直接恢复 ，可以使用的代码
    checkpoint = torch.load(model_save_path)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    """
    # 把模型整到gpu上 因为不管怎么样 我们都是从硬盘先读到cpu
    model = model.to(device)

    # 定义一些评价指标的列表来记录每个epoch模型的进步 也为了后面我们将这些指标可视化
    # 每个epoch验证集的准确率,loss放在这里
    val_acc_history = []
    val_losses = []
    # 每个epoch训练的准确率,loss放在这里
    train_acc_history = []
    train_losses = []

    # 深复制一下现在的model的权重当作最好model用来和后面训练的结果进行对比 好的话替换之
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(num_epoch):
        print('Epoch: {}/{}'.format(epoch+1,num_epoch))
        print('*'*10)

        # 开始训练和验证
        for phase in ['train','valid']:
            if phase == 'train':
                print('The training of epoch {} start'.format(epoch))
                model.train() # 进入训练模式
            if phase == 'valid':
                print('The validation of epoch {} start'.format(epoch))
                model.eval() # 进入评估模式

            epoch_loss = 0
            epoch_acc = 0

            # 开始一个batch一个batch地喂数据
            for _, sample in enumerate(dataloaders[phase]):
                inputs = sample['input']
                labels = sample['label']
                # 将cpu上的tensor搞到gpu上来
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # 将优化器的梯度清零
                # 只有训练的时候才会更新梯度所以这里我们需要灵活的将梯度追踪开关
                with torch.set_grad_enabled(phase == 'train'):
                    ouputs = model(inputs)
                    losses = criterion(ouputs,labels)

                    if phase == 'train':
                        losses.backward()
                        optimizer.step()
                # 计算这个batch的loss加到epoch的loss上
                epoch_loss += losses.item() * inputs.size(0) # 这里为什么这样写 因为 一个batch 如果有n割数据 这里的loss其实算的是平均
                # 按照标准的办法来计算准确率 pathnet的输出是相似度 所以用1减去loss的L2范数正则化 就是除以照片的大小
                epoch_acc += (1 - losses.item()/(224*224)) * inputs.size(0)
            # 到这 这个epoch的训练/验证就结束了 该有的数据都有了
            epoch_acc = epoch_acc/len(dataloaders[phase].dataset)
            epoch_loss = epoch/len(dataloaders[phase].dataset)

            # 算一下到现在的时间 并把开始时间归零
            time_consumed = time.time() - start_time
            print('Time consumed: {:.0f} m {:.0f} s'.format(time_consumed//60,time_consumed%60))
            print('{} Loss: {:.4f}   ACC: {:.4f}'.format(phase,epoch_loss,epoch_acc))

            # 下面要把数据加入到相应的列表（45 -51行的那些） 然后评判一下验证集中的准确率是否比最好的准确率还要好 如果是的话 那么我们就要保存一下这个模型
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
            if phase == 'valid':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())
                    state = {
                        'state_dict':model.state_dict(),
                        'best_acc':best_acc,
                        'optimizer':optimizer.state_dict(),
                    }
                    torch.save(state,model_save_path)
                val_acc_history.append(epoch_acc)
                val_losses.append(epoch_loss)
                scheduler.step() # 每个epoch如果到了这一步说明应该判断一下是否更新一下学习率了

        print('------------------------------------------------------------------')

    # 到这里训练完成
    time_consumed = time.time() - start_time
    print('Training Complete!  \nTime consumed: {:.0f} m {:.0f} s'.format(time_consumed // 60, time_consumed % 60))
    print('The Best Acc is {:.4f}'.format(best_acc))

    # 训练完我们应该返回一些值给用户用 比如那些评价指标之类的
    # 但首先 现在的这个model可能并不是最好准确率的那次 所以我们加载一下最好的那次
    model.load_state_dict(best_model_weights)
    return model, val_acc_history, val_losses, train_acc_history, train_losses















if __name__ == '__main__':
    '''
    以下是一个使用案例 这里我们就尽量简单
    '''
    # 构建数据集
    trainset = dataset_module.PN_dataset(transform=transforms.ToTensor())
    validset = dataset_module.PN_dataset(mode='valid_pathnet', transform=transforms.ToTensor())
    # 做一个加载器 用来按batch给网络喂数据
    trainloader = DataLoader(trainset, batch_size=CONST_BATCH_SIZE, shuffle=True, num_workers=4)
    validloader = DataLoader(validset, batch_size=CONST_BATCH_SIZE, shuffle=True, num_workers=4)
    loader_dict = {'train': trainloader, 'valid': validloader}  # 做一个字典  用来切换

    # 初始化网络
    net = net_modules.PathNet()
    # 初始化 loss计算标准
    criterion = nn.MSELoss()
    # 初始化一个Adam优化器 带动量
    optimizer = optim.Adam(net.parameters(), lr=CONST_LR)
    # 初始化一个学习率衰减器
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5, gamma=0.05) # 每5个epoch 调整一下学习率

    model, val_acc_history, val_losses, train_acc_history, train_losses = func_train_path(model=net,
                                                                                          dataloaders=loader_dict,
                                                                                          criterion=criterion,
                                                                                          optimizer=optimizer,
                                                                                          scheduler=scheduler,
                                                                                          model_save_path='test.pth',
                                                                                          num_epoch=20)
    print('TEST')
