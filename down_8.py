# -*- coding: utf-8 -*-
"""根据搜索词下载百度图片"""
import re
import sys
import urllib
import requests
import os

from tqdm import trange  # 显示进度条
from multiprocessing import cpu_count  # 查看cpu核心数
from multiprocessing import Pool  # 并行处理必备，进程池

headers = {

    "Host": "img5.artron.net",
    "Connection": "keep-alive",
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.91 Safari/537.36",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Referer": "http://auction.artron.net/paimai-art5113610001/",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.8",

}

def get_List_sub(List_imgs, num_cores):
    Len_imgs = len(List_imgs)  # 数据集长度
    if num_cores == 1:
        return List_imgs

    if num_cores == 2:  # 双核，将所有数据集分成两个子数据集
        subset1 = List_imgs[:Len_imgs // 2]
        subset2 = List_imgs[Len_imgs // 2:]

        List_subsets = [subset1, subset2]
    elif num_cores == 4:  # 四核，将所有数据集分成四个子数据集
        subset1 = List_imgs[:Len_imgs // 4]
        subset2 = List_imgs[Len_imgs // 4: Len_imgs // 2]
        subset3 = List_imgs[Len_imgs // 2: (Len_imgs * 3) // 4]
        subset4 = List_imgs[(Len_imgs * 3) // 4:]

        List_subsets = [subset1, subset2, subset3, subset4]
    elif num_cores >= 8:  # 八核以上，将所有数据集分成八个子数据集
        num_cores = 8
        subset1 = List_imgs[:Len_imgs // 8]
        subset2 = List_imgs[Len_imgs // 8: Len_imgs // 4]
        subset3 = List_imgs[Len_imgs // 4: (Len_imgs * 3) // 8]
        subset4 = List_imgs[(Len_imgs * 3) // 8: Len_imgs // 2]
        subset5 = List_imgs[Len_imgs // 2: (Len_imgs * 5) // 8]
        subset6 = List_imgs[(Len_imgs * 5) // 8: (Len_imgs * 6) // 8]
        subset7 = List_imgs[(Len_imgs * 6) // 8: (Len_imgs * 7) // 8]
        subset8 = List_imgs[(Len_imgs * 7) // 8:]

        List_subsets = [subset1, subset2, subset3, subset4,
                        subset5, subset6, subset7, subset8]
    return List_subsets

def down_pic(lines, pic_save):
    for line in lines:
        words = line.split('\t')
        person = words[0]
        num = words[1]
        pic_url = words[2]
        rects = words[3].split(',')
        md5sum = words[4].rstrip('\n')

        try:
            pic = requests.get(pic_url, timeout=15, header=headers)
            pic_type = pic_url.split('.')[-1]
            pic_name = pic_save + '/' + person +'_'+ num + '.'+pic_type
            with open(pic_name, 'wb') as f:
                f.write(pic.content)
                print("dowmload {}".format(person))

        except Exception as e:
            print(pic_url)
            print(e)


if __name__ == '__main__':
    pic_save = 'download_face'
    if not os.path.exists(pic_save):
        os.mkdir(pic_save)

    with open('util.txt', 'r') as file:
        lines=file.readlines()

    # 开辟进程池，num_cores为cpu核心数，也就是开启的进程数
    num_cores = 2#cpu_count()  # cpu核心数
    List_subsets = get_List_sub(lines, num_cores)
    p = Pool(num_cores)

    for i in range(num_cores):
        # p.apply(func=down_pic, args=(List_subsets[i], pic_save, ))
        p.apply_async(func=down_pic, args=(List_subsets[i], pic_save, ))


