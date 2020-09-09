# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : split_data.py
# @Software: PyCharm
import random


def split_data(full_list, shuffle=False, ratio=0.2):
    """
    分割数据集
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    train_data = full_list[offset:]
    dev_data = full_list[:offset]
    return train_data, dev_data