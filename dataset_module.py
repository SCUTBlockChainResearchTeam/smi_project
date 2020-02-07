from torch.utils.data import Dataset
from load_data import load_csv
import pandas as pd
import numpy as np
from PIL import Image
"""
构建我们自己的dataset  这里需要重载以下三个函数
"""

'''
详细解释一下path_net数据集的架构
最终的目标是输入是224*224*2  输出是 224*224 
一开始我们只有一个csv表格 和一些照片 参见./train_pathnet
csv表格的结构是  image_path(str) || target_point(tuple) || path(array with shape 2 * n)
分别的含义是   读取一张照片的地址(用来构建输入的第一个频道) || 我们想要训练的点（构建第二个频道） || 刚才那个点在ground truth上所在的路径
'''
class PN_dataset(Dataset):
    # @param:
    def __init__(self, mode='train_pathnet', transform=None):
        self.data_size, self.df= load_csv(mode)
        self.root_dir = mode
        self.transform = transform
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        # 把照片取出来转为 ndarray 作为第一个输入频道
        img = Image.open(img_path, 'r')
        channel_1 = np.array(img)
        # 现在把坐标点提取出来构建第二个输入频道
        x_pos, y_pos = eval(self.df.iloc[idx,1])
        channel_2 = np.zeros_like(channel_1)
        channel_2[x_pos,y_pos] = 1# 这里我们先随便设置一个值
        # 合并两个channel 成为我们的input
        input_ = np.array([channel_1, channel_2])
        input_ = input_.transpose(1,2,0)
        # 下面通过最后的path来构建出与上面的点在一条轨迹上的图片的label
        label_ = np.zeros_like(channel_2)
        x_label, y_label = eval(self.df.iloc[idx,2])[0], eval(self.df.iloc[idx,2])[1]
        label_[x_label,y_label] = 1 # 也一样把这些target暂都设为1
        sample = {'input':input_, 'label':label_}
        if self.transform:
            sample['input'] = self.transform(sample['input'])
            sample['label'] = self.transform(sample['label'])
        return sample



'''
下面构建overlap_net的数据集
overlap_net 的输入是单通道的 224*224 输出也是224*224 输出上是我们估计的重合区域
和path_net一样 也是csv文件和照片 但是cvs的结构是这样的 : image_path(str) || X_label(array) || Y_label(array)
 X_label: 重合区域的x坐标们
 Y_label: 重合区域的纵坐标们
'''
class ON_dataset(Dataset):
    def __init__(self,mode = 'train_overlapnet', transform=None):
        self.data_size, self.df = load_csv(mode)
        self.root_dir = mode
        self.transform = transform

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx,0]
        input_ = np.array(Image.open(img_path)) # 嵌套这基层是将 input变为h*w*c的形式
        # 下面构建label
        x_label = eval(self.df.iloc[idx,1])
        y_label = eval(self.df.iloc[idx,2])
        label = np.zeros_like(input_)
        label[x_label,y_label] = 1
        if self.transform:
            input_ = self.transform(input_)
            label = self.transform(label)
        sample = {'input': input_, 'label': label}
        return sample