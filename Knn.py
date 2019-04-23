import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
path = os.path.abspath('..')
if not path in sys.path:
    sys.path.append(path)

def binaryzation(data):
    ret = np.empty((data.shape[0]))
    for i in range(data.shape[0]):
        ret[i] = 0
        if (data[i] > 127):
            ret[i] = 1
    return ret

def load_data(dataset='training'):
    filepath = './MNIST_data/train.csv'
    datas = pd.read_csv(filepath).values
    datas, labels = datas[1:, 1:], datas[1:, 0]
    idx = np.random.permutation(datas.shape[0])
    fileInd = []
    for i in range(datas.shape[0]):
        fileInd.append(idx[i])

    data = np.empty((datas.shape[0], 28 * 28), dtype="float32")
    label = np.empty((datas.shape[0]), dtype="uint8")
    print("loading data...")
    for c in range(datas.shape[0]):
        tmpInd = fileInd[c]
        data[c] = binaryzation(datas[tmpInd])
        label[c] = labels[tmpInd]

    return data, label




    # if dataset == 'training':
    #     filepath = './MNIST_data/train.csv'
    # elif dataset == 'testing':
    #     filepath = './MNIST_data/test.csv'
    # else:
    #     print("dataset type error")
    #     return None
    # traindata = pd.read_csv(filepath).values
    # datas, labels = traindata[1:, 1:], traindata[1:, 0]
    # datas = np.array(datas)
    # labels = np.array(labels)
    # # labels = np_utils.to_categorical(labels, 10)
    # return datas, labels

def KNN(test_vec, train_data, train_label, k):
    train_data_size = train_data.shape[0]
    dif_mat = np.tile(test_vec, (train_data_size, 1)) - train_data
    sqr_dif_mat = dif_mat ** 2
    sqr_dis = sqr_dif_mat.sum(axis=1)

    sorted_idx = sqr_dis.argsort()

    class_cnt = {}
    maxx = 0
    best_class = 0
    for i in range(k):
        tmp_class = train_label[sorted_idx[i]]
        tmp_cnt = class_cnt.get(tmp_class, 0) + 1
        class_cnt[tmp_class] = tmp_cnt
        if (tmp_cnt > maxx):
            maxx = tmp_cnt
            best_class = tmp_class
    return best_class


if __name__ == "__main__":
    train_img, train_lbl = load_data(dataset='training')
    test_img, test_lbl = load_data(dataset='testing')

    # im = Image.fromarray(np.uint8(train_img[0].reshape(28,28)))
    # im.show()

    # np.random.seed(123456)
    tot = test_img.shape[0]
    err = 0
    print("testing...")
    for i in range(tot):
        best_class = KNN(test_img[i], train_img, train_lbl, 3)
        print(i, "/", tot, "\r", best_class)
        if (best_class != test_lbl[i]):
            err = err + 1.0
    print("accuracy")
    print(1 - err / tot)

# accuracy
# 0.9819281411462177
