from torch.utils.data import Dataset
from load_data import load_csv
import pandas as pd
import numpy as np
from PIL import Image
"""
构建我们自己的dataset  这里需要重载以下三个函数
"""


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



