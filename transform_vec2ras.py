"""
 A tool suite to transform the vector representation
 of images into raster form. Here are some parameter
 @:param:  vector_images: list of images
           side: size after trans

"""


import cairo
import numpy as np


def vector_to_raster(vector_image, side=28, line_diameter=1, padding=0, bg_color=(0, 0, 0), fg_color=(1, 1, 1)):
    """
    padding and line_diameter are relative to the original 256x256 image.
    """

    original_side = 256.
    # 从这里开始把pycairo具体的使用解析一下
    '''
    这个类就是生产image用的对象 需要的参数里第一个比较重要 
    他是pycairo自身定义的常量 代表32位保存一个像素 透明度以及RGB各占8位
    '''
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)

    '''
    以下只不过是一些基本设置 
    '''
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    # 设置好画图的画板尺寸 这里是正则化的过程  也就说  原先我们的大小是 256+padding大小 但是我们要映射到256 那么这里就有一个比例
    ctx.scale(new_scale, new_scale)
    # 改变transformation 矩阵 确定从哪里开始绘制
    ctx.translate(total_padding / 2., total_padding / 2.)
    # clear background
    ctx.set_source_rgb(*bg_color)
    ctx.paint()

    # 这里粗略理解实在调整边界的位置 其实就是要居中
    bbox = np.hstack(vector_image).max(axis=1)  # 找到这张图像右下角
    offset = ((original_side, original_side) - bbox) / 2.  # 设置偏移量 看右下角和原始大小最右下角的差距 paddding肯定是要/2
    offset = offset.reshape(-1, 1)
    centered = [stroke + offset for stroke in vector_image]  # 这样就把坐标对齐到右下角了 但是因为只走了一半 相对来说就是居中

    # draw strokes, this is the most cpu-intensive part 比较耗费cpu的部分
    ctx.set_source_rgb(*fg_color)  # 先改变画笔的颜色 从黑到白  因为我们现在要开始划线了
    # 记住 centered和vector_image其实是一样的 形状： [((1,3,5,7,6),(55,42,12,16,99)),((),()),((),()),.....]
    for xv, yv in centered:
        ctx.move_to(xv[0], yv[0])
        for x, y in zip(xv, yv):
            ctx.line_to(x, y)
        ctx.stroke()
    data = surface.get_data()
    raster_image = np.copy(np.asarray(data)[::4])
    return raster_image


'''
  Why need this func:
    We can use this func to extract strokes from a single image
    ex: for image A, there are M vectorized path, these paths are
    parts of labels in TRAINING_SET and VALIDATION_SET 
'''


def extract_strokes(vector_image, side=28, line_diameter=1, padding=0, bg_color=(0, 0, 0), fg_color=(1, 1, 1)):
    original_side = 256.
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    '''
    我们还要设置一块区域  这个区域的用处是将每张图片里面的stroke画出来然后放到图片的列表里

    '''
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    # 设置好画图的画板尺寸 这里是正则化的过程  也就说  原先我们的大小是 256+padding大小 但是我们要映射到256 那么这里就有一个比例
    ctx.scale(new_scale, new_scale)
    # 改变transformation 矩阵 确定从哪里开始绘制
    ctx.translate(total_padding / 2., total_padding / 2.)

    vec_strokes = []

    # 这里粗略理解实在调整边界的位置 其实就是要居中
    bbox = np.hstack(vector_image).max(axis=1)  # 找到这张图像右下角
    offset = ((original_side, original_side) - bbox) / 2.  # 设置偏移量 看右下角和原始大小最右下角的差距 paddding肯定是要/2
    offset = offset.reshape(-1, 1)
    centered = [stroke + offset for stroke in vector_image]  # 这样就把坐标对齐到右下角了 但是因为只走了一半 相对来说就是居中
    for xv, yv in centered:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()
        # draw strokes, this is the most cpu-intensive part 比较耗费cpu的部分
        ctx.set_source_rgb(*fg_color)  # 先改变画笔的颜色 从黑到白  因为我们现在要开始划线了
        ctx.move_to(xv[0], yv[0])
        for x, y in zip(xv, yv):
            ctx.line_to(x, y)
        ctx.stroke()
        data1 = surface.get_data()
        vecor_stroke = np.copy(np.asarray(data1)[::4])
        vec_strokes.append(vecor_stroke)
    return vec_strokes


if __name__ == '__main__':
    print('TEST')

