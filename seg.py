import numpy as np
import matplotlib
import os
from PIL import Image

def img_seg(dir):
    files = os.listdir(dir)
    for file in files:
        a, b = os.path.splitext(file)
        img = Image.open(os.path.join(dir + "/" + file))
        hight, width = img.size
        w = 500    #切割成812*812
        id = 1
        i = 0
        while (i + w <= hight):
            j = 0
            while (j + w <= width):
                new_img = img.crop((i, j, i + w, j + w))
                rename = "../rock/pic_new/"
                new_img.save(rename + a + "_" + str(id) + b)
                id += 1
                j += 500   #滑动步长
            i = i + 500


path = '../rock/train'
img_seg(path)
