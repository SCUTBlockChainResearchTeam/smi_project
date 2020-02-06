import struct
from struct import unpack
import os
import transform_vec2ras
import csv
from PIL import Image
import time
import numpy as np
import pandas as pd
import cv2
"""
 read csv file from indicated dir 
 this csv is our source information to construct TRAIN_SET
"""


def load_csv(paths):
    df = pd.read_csv(os.path.join(paths,'data.csv'))
    return df.shape[0], df

'''
read image information from .bin file  
'''
def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = list(unpack(fmt, file_handle.read(n_points)))
        y = list(unpack(fmt, file_handle.read(n_points)))
        image.append([x, y])
    return image


def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f) # 一次返回一张图片的向量列表
            except struct.error:
                break


'''
 init the pictures and csv file for training
 @:param: path_name : path of the .bin files 
          num: extract how many picture from every bin file
          size: the size of picture we want to extract
'''


def init_training_pathnet(path_name,num,size = 224, mode='train_pathnet'):
    with open(os.path.join(mode,'data.csv'), 'w') as f:
        writer = csv.DictWriter(f,['image_name','pixel_pos','located_path'])
        writer.writeheader()
        # start our procedure from .bin files
        files = os.listdir(path_name)
        for file in files:
            i = 0
            for image in unpack_drawings(os.path.join(path_name,file)):
                # 接下来的过程是 一张照片 做成多个样本
                raster_img = transform_vec2ras.vector_to_raster(image, side=size)
                # 把这张照片保存
                raster_img.resize(size,size)
                raster_img = Image.fromarray(raster_img)
                img_path = mode + '\\' + ''.join(str(time.time()).split('.')) + '.png'
                raster_img.save(img_path)
                # 接下来比较复杂 我们需要把每条线条提取一下 然后这条线条上的点  每一个都是一个简单的sample
                for stroke in transform_vec2ras.extract_strokes(image,side=size):
                    # 先把stroke reshape
                    stroke.resize(size,size)
                    x_label , y_label = np.nonzero(stroke)

                    for point in zip(x_label, y_label):
                        # 初始化一个dict用来按行存放数据
                        csv_dict = {}
                        csv_dict['image_name'] = img_path  # 照片存储的地方
                        csv_dict['pixel_pos'] = point  # 一个元组
                        csv_dict['located_path'] = [list(x_label), list(y_label)]  # 一个列表里面放了两个列表 x y
                        writer.writerow(csv_dict) # 存一行进去
                i += 1
                if i == num:
                    break

'''
/////init training data for overlap_net part. In this part we focus on finding the overlap region
in a picture. Our direct thinking is: 
     first: reconstruct a full image from vectorized pic

@:param
        pathname: directory of .bin file
        
@:return none


'''

def init_training_overlapnet(pathname,num=1,pic_size = 224,  mode='train_overlapnet'):
    # 每次运行函数之前先检查.csv文件是否存在 如果存在的话 那么应该先删掉再进行这个操作
    with open(os.path.join(mode,'data.csv'),'w') as f:
        writer = csv.DictWriter(f,['image_name','X_label','Y_label'])
        writer.writeheader()
        # 现在有了表头，就可以开始一张图片一张图片的处理了
        # 先列出 pathname下所有的文件
        files = os.listdir(pathname)
        for file in files:
            count = 0
            for image in unpack_drawings(os.path.join(pathname, file)):
                overlaps_x = []
                overlaps_y = []
                img_path = mode + '\\' + ''.join(str(time.time()).split('.')) + '.png'
                raster_img = transform_vec2ras.vector_to_raster(image,pic_size) # 获取到未重整的位图数据
                raster_img.resize(pic_size,pic_size)
                img = Image.fromarray(raster_img)
                img.save(img_path) # 到这里 我们就把这张照片存到了对应的文件夹里

                # 先把每条单独的路径搞成 a*a 的样子 然后放到一个数组里面我们备用
                strokes = [np.uint8(stroke.reshape(pic_size,pic_size)) for stroke in transform_vec2ras.extract_strokes(image,side=pic_size)]
                strokes_ = []
                for stroke in strokes:
                    res,stroke_ = cv2.threshold(stroke,1,255,cv2.THRESH_BINARY)
                    strokes_.append(stroke_)


                # 然后两两对比找到相交的部分
                for i in range(len(strokes_)):
                    for j in range(i+1,len(strokes_)):
                        overlap_region = cv2.bitwise_and(strokes_[i],strokes_[j])
                        x , y = np.nonzero(overlap_region)
                        overlaps_x = np.hstack([overlaps_x,x])
                        overlaps_y = np.hstack([overlaps_y, y])

                # 现在关于这幅图像我们需要的东西都拿到了 构建 存储到csv中的行字典
                csv_dict = {}
                csv_dict['image_name'] = img_path
                csv_dict['X_label'] = list(overlaps_x)
                csv_dict['Y_label'] = list(overlaps_y)
                writer.writerow(csv_dict)
                # 将计数变量递增 以便提取到我们想要的图片数量
                count += 1
                if count == num:
                    break

if __name__ == '__main__':
    print('TEST')
    init_training_overlapnet('./imagesource',5)


