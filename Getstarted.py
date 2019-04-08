import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import pandas as pd
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras import callbacks
from sklearn.model_selection import train_test_split
from keras.models import Sequential, model_from_yaml, load_model
from keras.applications.resnet50 import preprocess_input, decode_predictions
# import csv

filepath = './MNIST_data/train.csv'
# with open(filepath) as f:
#     reader = csv.DictReader(f)
#     datas=[]
#     labels =[]
#     for row in reader:
#         datas.append(row[0])
#         labels.append(row[2:])

filepathT = './MNIST_data/test.csv'
# with open(filepath) as f:
#     reader = csv.DictReader(f)
#     datast=[]
#     labelst =[]
#     for row in reader:
#         datast.append(row[0])
#         labelst.append(row[2:])
traindata = pd.read_csv(filepath).values
datas ,labels= traindata[1:,1:],traindata[1:,0]
# traindataT = pd.read_csv(filepathT).values
# datast ,labelst= traindataT[1:,1:],traindataT[1:,0]


data = np.array(datas)
labels = np.array(labels)
# datat = np.array(datast)
# labelst = np.array(labelst)

labels = np_utils.to_categorical(labels, 10)
# labelst = np_utils.to_categorical(labelst, 10)

input_shape  = (28, 28, 1)
#
batch_size = 128
nb_classes = 10
nb_epoch = 12
#
# # 输入图像的维度，此处是mnist图像，因此是28*28
img_rows, img_cols = 28, 28
# # 卷积层中使用的卷积核的个数
nb_filters = 32
# # 池化层操作的范围
pool_size = (2,2)
# # 卷积核的大小
kernel_size = (3,3)
# # keras中的mnist数据集已经被划分成了60,000个训练集，10,000个测试集的形式，按以下格式调用即可
x_train, x_test, y_train, y_test = train_test_split(datas, labels, test_size=0.2)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)


K.set_session(sess)
img = tf.placeholder(tf.float32, shape=(None, 784))


# 建立序贯模型
model = Sequential()

# 卷积层，对二维输入进行滑动窗卷积
# 当使用该层为第一层时，应提供input_shape参数，在tf模式中，通道维位于第三个位置
# border_mode：边界模式，为"valid","same"或"full"，即图像外的边缘点是补0
# 还是补成相同像素，或者是补1
model.add(Convolution2D(nb_filters, kernel_size[0] ,kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))

# 卷积层，激活函数是ReLu
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))

# 池化层，选用Maxpooling，给定pool_size，dropout比例为0.25
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

# Flatten层，把多维输入进行一维化，常用在卷积层到全连接层的过渡
model.add(Flatten())

# 包含128个神经元的全连接层，激活函数为ReLu，dropout比例为0.5
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 包含10个神经元的输出层，激活函数为Softmax
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# 输出模型的参数信息
model.summary()
# 配置模型的学习过程
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


print("train.......")
tbCallbacks = callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(x_test, y_test), callbacks=[tbCallbacks])
scroe, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
print('scroe:', scroe, 'accuracy:', accuracy)
yaml_string = model.to_yaml()
with open('./models/Mnist.yaml', 'w') as outfile:
    outfile.write(yaml_string)
model.save_weights('./models/Mnist.h5')

def pred_data():

    with open('./models/Mnist.yaml') as yamlfile:
        loaded_model_yaml = yamlfile.read()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights('./models/Mnist.h5')

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # images = []
    filepathT = './MNIST_data/test.csv'
    traindataT = pd.read_csv(filepathT).values
    datast = traindataT[1:,:]
    data = np.array(datast)
    for c in data:
        x = np.expand_dims(c, axis=0)
        x = preprocess_input(x)
        result = model.predict_classes(x,verbose=0)

        print(c,result[0])