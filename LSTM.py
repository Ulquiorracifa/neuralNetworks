
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib import rnn
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
def model(X, W, B, lstm_size):
    time_step_size = 28  # 循环层长度



    # X, input shape: (batch_size, time_step_size, input_vec_size)
    #  XT shape: (time_step_size, batch_size, input_vec_size)
    # #对这一步操作还不是太理解，为什么需要将第一行和第二行置换
    XT = tf.transpose(X, [1, 0, 2]) # permute time_step_size and batch_size,[28, 128, 28]

    #  XR shape: (time_step_size * batch_size, input_vec_size)
    XR = tf.reshape(XT, [-1, lstm_size]) # each row has input for each lstm cell (lstm_size=input_vec_size)

    #  Each array shape: (batch_size, input_vec_size)
    X_split = tf.split(XR, time_step_size, 0) # split them to time_step_size (28 arrays),shape = [(128, 28),(128, 28)...]

    #  Make lstm with lstm_size (each input vector size). num_units=lstm_size; forget_bias=1.0
    lstm = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)

    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    #  rnn..static_rnn()的输出对应于每一个timestep，如果只关心最后一步的输出，取outputs[-1]即可
    outputs, _states = rnn.static_rnn(lstm, X_split, dtype=tf.float32) # 时间序列上每个Cell的输出:[... shape=(128, 28)..]
    #  tanh activation # Get the last output
    return tf.matmul(outputs[-1], W) + B, lstm.state_size # State size to initialize the state


if __name__ == "__main__":
    filepath = './MNIST_data/train.csv'
    traindata = pd.read_csv(filepath).values
    datas, labels = traindata[1:, 1:], traindata[1:, 0]
    data = np.array(datas)
    labels = np.array(labels)
    labels = np_utils.to_categorical(labels, 10)
    x_train, x_test, y_train, y_test = train_test_split(datas, labels, test_size=0.2)

    img_rows, img_cols = 28, 28
    input_vec_size = lstm_size = 28  # 输入向量的维度
    time_step_size = 28  # 循环层长度

    batch_size = 128
    test_size = 256


    x_train = x_train.reshape(-1, img_rows, img_cols)
    x_test = x_test.reshape(-1, img_rows, img_cols)

    X = tf.placeholder(tf.float32, [None, 28, 28])
    Y = tf.placeholder(tf.float32, [None, 10])

    W = init_weights([lstm_size, 10])  # 输出层权重矩阵28×10
    B = init_weights([10])  # 输出层bais



    py_x, state_size = model(X, W, B, lstm_size)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    # 返回每一行的最大值
    predict_op = tf.argmax(py_x, 1)

    # tf.ConfigProto，一般是在创建session时对session进行配置
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True  # 允许gpu在使用的过程中慢慢增加。

    with tf.Session(config=session_conf) as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        for i in range(100):
            # 从训练集中每段选择一个batch训练，batch_size= end-start
            for start, end in zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train) + 1, batch_size)):
                sess.run(train_op, feed_dict={X: x_train[start:end], Y: y_train[start:end]})
                # X (128,28,28)
            s = len(x_test)
            test_indices = np.arange(len(x_test))  # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]

            print(i, np.mean(np.argmax(y_test[test_indices], axis=1) == sess.run(predict_op, feed_dict={X: x_test[test_indices]})))