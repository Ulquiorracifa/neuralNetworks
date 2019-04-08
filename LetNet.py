import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras import callbacks
from sklearn.model_selection import train_test_split
from keras.models import Sequential, model_from_yaml, load_model
from keras.applications.resnet50 import preprocess_input, decode_predictions





def LeNet():
    model = Sequential()
    model.add(Conv2D(32,(5,5),strides=(1,1),input_shape=(28,28,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    return model

if __name__ == "__main__":
    filepath = './MNIST_data/train.csv'
    traindata = pd.read_csv(filepath).values
    datas, labels = traindata[1:, 1:], traindata[1:, 0]
    data = np.array(datas)
    labels = np.array(labels)
    labels = np_utils.to_categorical(labels, 10)
    input_shape = (28, 28, 1)
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
    pool_size = (2, 2)
    # # 卷积核的大小
    kernel_size = (3, 3)
    # # keras中的mnist数据集已经被划分成了60,000个训练集，10,000个测试集的形式，按以下格式调用即可
    x_train, x_test, y_train, y_test = train_test_split(datas, labels, test_size=0.2)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    K.set_session(sess)
    img = tf.placeholder(tf.float32, shape=(None, 784))

    model= LeNet()

    # 输出模型的参数信息
    model.summary()
    # 配置模型的学习过程
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    print("train.......")
    tbCallbacks = callbacks.TensorBoard(log_dir='./logs/logsLetNet', histogram_freq=1, write_graph=True, write_images=True)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(x_test, y_test),
              callbacks=[tbCallbacks])
    scroe, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('scroe:', scroe, 'accuracy:', accuracy)
    yaml_string = model.to_yaml()
    with open('./models/MnistLetNet.yaml', 'w') as outfile:
        outfile.write(yaml_string)
    model.save_weights('./models/MnistLetNet.h5')