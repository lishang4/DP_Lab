# -*- coding: UTF-8 -*-
'''
created at 2020/11/6
author: Lishang Chien
'''
import numpy as np
from os.path import abspath, join


def load_img(path):
    """load train_img/test_img txt file"""
    data = []
    try:
        with open(f"{abspath('.')}{path}", 'r') as txtFile:
            for line in txtFile:
                line = line.replace('\n', '').split(',')
                x = [[float(_)] for _ in line]
                data.append(np.array(x))
        return data
    except Exception as e:
        raise

def load_label(path, one_hot_vector=False):
    """load train_label/test_label txt file"""
    data = []
    try:
        with open(f"{abspath('.')}{path}", 'r') as txtFile:
            for line in txtFile:
                x = line.replace('\n', '').split(',')
                data.append(np.int64(x[0]))
        return data
    except Exception as e:
        raise

def one_hot_vector(label, data=[]):
    for x in label:
        if x == 0:
            x = [[1.], [0.], [0.]]
        elif x == 1:
            x = [[0.], [1.], [0.]]
        elif x == 2:
            x = [[0.], [0.], [1.]]
        data.append(np.array(x))
    return data

def load_data_wrapper():
    # load file
    print(f'loading data ...')
    tr_d = load_img('/data/train_img.txt')
    tr_l = load_label('/data/train_label.txt')
    td_d = load_img('/data/test_img.txt')

    # grouping
    print(f'normalization data ...')
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[:6000]]
    training_data = list(zip(training_inputs, one_hot_vector(tr_l[:6000])))

    validation_inputs = [np.reshape(x, (784, 1)) for x in tr_d[6000:]]
    validation_data = list(zip(validation_inputs, tr_l[6000:]))

    test_inputs = [np.reshape(x, (784, 1)) for x in td_d]
    test_data = list(test_inputs)

    return training_data, validation_data, test_data

def output(path, data):
    try:
        with open(f"{abspath('.')}{path}", 'w+') as txtFile:
            for answer in data:
                txtFile.write(f'{str(answer)}\n')
    except Exception as e:
        raise