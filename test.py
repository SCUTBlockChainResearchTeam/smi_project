import numpy as np
import torchvision.transforms as tsfm
import torchvision.models as models
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import *
import random
import os
import time
import torch

# 3.12 1-max问题

# n = 30
# iter_num = 1000
# scale = 20  # 种群规模
# p = 0.1  # 变异率
#
# parents = []
# children = []
# # 先生成初代20个个体
# def generate_group():
#     parents.clear()
#     scale_num = pow(2, n)
#     for i in range(20):
#         chosen = random.randint(0, scale_num)
#         chosen_bin = list(bin(chosen).replace('0b', '').rjust(30, '0'))
#         chosen_digit = list(map(int, chosen_bin))
#         parents.append(chosen_digit)
#
#
# def choose_parent(parent_list):
#     sum_of_fitness = sum(cal_fitness(parent) for parent in parent_list)
#     r = random.uniform(0,sum_of_fitness)
#     F = cal_fitness(parent_list[0])
#     k = 0
#     while F < r:
#         k += 1
#         F += cal_fitness(parent_list[k])
#
#     return parent_list[k]
#
#
# def cal_fitness(parent):
#     return sum(parent)
#
#
# def variation(possibility,oringin):
#     for i in range(len(oringin)):
#         r = random.uniform(0,1)
#         if r < possibility:
#             if oringin[i] == 0:
#                 oringin[i] = 1
#             else:
#                 oringin[i] = 0
#     return oringin
#
#
# def mating(parent1,parent2,length):
#     cp_1 = list.copy(parent1)
#     cp_2 = list.copy(parent2)
#     pos_of_exchange = random.randint(1, length-1)
#     for i in range(pos_of_exchange,len(parent1)):
#         parent1[i] = cp_2[i]
#         parent2[i] = cp_1[i]
#     return parent1,parent2
#
#
# def statistic(parent_list):
#     avg = sum(sum(parent) for parent in parent_list)/len(parent_list)
#     min_ = min(sum(parent) for parent in parent_list)
#     return avg , min_
#
# if __name__ == '__main__':
#     generate_group()
#     if len(parents) > 0:
#         print('种群生成成功')
#         x = [k for k in range(1,1001)]
#         avg_list = []
#         min_list = []
#         for i in range(iter_num):
#             children.clear()
#             while len(children) < len(parents):
#                 parent1 = choose_parent(parents)
#                 parent2 = choose_parent(parents)
#                 parent1,parent2 = mating(parent1,parent2,n)
#                 parent1 = variation(p,parent1)
#                 parent2 = variation(p, parent2)
#                 children.append(parent1)
#                 children.append(parent2)
#             parents = list.copy(children)
#             avg,min_ = statistic(parents)
#             avg_list.append(avg)
#             min_list.append(min_)
#             print('第{}次迭代后平均适应度为：{}  最好的个体适应度：{}'.format(str(i) , str(avg) , str(min_)))


# 3.13 连续遗传最小化sphere函数

# 先熟练一下作图的步骤 尤其3D作图 拿Sphere函数练手

# 指定绘图轴为3D绘图
# figure = plt.figure()
# ax = plt.axes(projection = '3d')
#
# # 和二维图不一样 三维图需要准备一下数据
# x = np.linspace(-5,5,100)
# y = np.linspace(-5,5,100)
# z = x**2 + y**2
# X, Y = np.meshgrid(x,y) #这一步很重要
# Z = X**2 + Y**2

# 绘制三维散点图
# ax.scatter3D(X,Y,Z,cmap = 'Blues')
# 绘制三维曲线使用plot3D 下面是绘制三维曲面
# ax.plot_surface(X,Y,Z,cmap='rainbow')
# ax.contour(X,Y,Z,zdim = 'z',offset = 0 , cmap = 'Blues') #等高线图
# plt.show()

if __name__ == '__main__':
    a = torch.from_numpy(np.array([[1,2,3],[4,5,6]]))
    c = a > 3
    c.type(torch.threshold())
    print(type(c),c)

