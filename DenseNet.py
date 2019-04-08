import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, Input, Flatten
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.layers.merge import concatenate

from keras.utils import np_utils
from keras import backend as K
from keras import callbacks
from sklearn.model_selection import train_test_split
from keras.models import Sequential, model_from_yaml, load_model
from keras.applications.resnet50 import preprocess_input, decode_predictions

#
# def DenseNet121(nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, classes=1000, weights_path=None):
#     '''Instantiate the DenseNet 121 architecture,
#         # Arguments
#             nb_dense_block: number of dense blocks to add to end
#             growth_rate: number of filters to add per dense block
#             nb_filter: initial number of filters
#             reduction: reduction factor of transition blocks.
#             dropout_rate: dropout rate
#             weight_decay: weight decay factor
#             classes: optional number of classes to classify images
#             weights_path: path to pre-trained weights
#         # Returns
#             A Keras model instance.
#     '''
#     eps = 1.1e-5
#
#     # compute compression factor
#     compression = 1.0 - reduction
#
#     # Handle Dimension Ordering for different backends
#     global concat_axis
#     if K.image_dim_ordering() == 'tf':
#       concat_axis = 3
#       img_input = Input(shape=(224, 224, 3), name='data')
#     else:
#       concat_axis = 1
#       img_input = Input(shape=(3, 224, 224), name='data')
#
#     # From architecture for ImageNet (Table 1 in the paper)
#     nb_filter = 64
#     nb_layers = [6,12,24,16] # For DenseNet-121
#
#     # Initial convolution
#     x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
#     x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
#     x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
#     x = Scale(axis=concat_axis, name='conv1_scale')(x)
#     x = Activation('relu', name='relu1')(x)
#     x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
#     x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)
#
#     # Add dense blocks
#     for block_idx in range(nb_dense_block - 1):
#         stage = block_idx+2
#         x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
#
#         # Add transition_block
#         x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
#         nb_filter = int(nb_filter * compression)
#
#     final_stage = stage + 1
#     x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
#
#     x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
#     x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
#     x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
#     x = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
#
#     x = Dense(classes, name='fc6')(x)
#     x = Activation('softmax', name='prob')(x)
#
#     model = Model(img_input, x, name='densenet')
#
#     if weights_path is not None:
#       model.load_weights(weights_path)
#
#     return model
#
#
# def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
#     '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
#         # Arguments
#             x: input tensor
#             stage: index for dense block
#             branch: layer index within each dense block
#             nb_filter: number of filters
#             dropout_rate: dropout rate
#             weight_decay: weight decay factor
#     '''
#     eps = 1.1e-5
#     conv_name_base = 'conv' + str(stage) + '_' + str(branch)
#     relu_name_base = 'relu' + str(stage) + '_' + str(branch)
#
#     # 1x1 Convolution (Bottleneck layer)
#     inter_channel = nb_filter * 4
#     x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
#     x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
#     x = Activation('relu', name=relu_name_base+'_x1')(x)
#     x = Convolution2D(inter_channel, 1, 1, name=conv_name_base+'_x1', bias=False)(x)
#
#     if dropout_rate:
#         x = Dropout(dropout_rate)(x)
#
#     # 3x3 Convolution
#     x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
#     x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
#     x = Activation('relu', name=relu_name_base+'_x2')(x)
#     x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
#     x = Convolution2D(nb_filter, 3, 3, name=conv_name_base+'_x2', bias=False)(x)
#
#     if dropout_rate:
#         x = Dropout(dropout_rate)(x)
#
#     return x
#
#
# def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
#     ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
#         # Arguments
#             x: input tensor
#             stage: index for dense block
#             nb_filter: number of filters
#             compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
#             dropout_rate: dropout rate
#             weight_decay: weight decay factor
#     '''
#
#     eps = 1.1e-5
#     conv_name_base = 'conv' + str(stage) + '_blk'
#     relu_name_base = 'relu' + str(stage) + '_blk'
#     pool_name_base = 'pool' + str(stage)
#
#     x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
#     x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
#     x = Activation('relu', name=relu_name_base)(x)
#     x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)
#
#     if dropout_rate:
#         x = Dropout(dropout_rate)(x)
#
#     x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)
#
#     return x
#
#
# def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
#     ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
#         # Arguments
#             x: input tensor
#             stage: index for dense block
#             nb_layers: the number of layers of conv_block to append to the model.
#             nb_filter: number of filters
#             growth_rate: growth rate
#             dropout_rate: dropout rate
#             weight_decay: weight decay factor
#             grow_nb_filters: flag to decide to allow number of filters to grow
#     '''
#
#     eps = 1.1e-5
#     concat_feat = x
#
#     for i in range(nb_layers):
#         branch = i+1
#         x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
#         concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))
#
#         if grow_nb_filters:
#             nb_filter += growth_rate
#
#     return concat_feat, nb_filter

def densenet(x):
    x1 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x2 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=(1, 1))(x1)

    x3 = concatenate([x1, x2], axis=3)
    x = BatchNormalization()(x3)
    x = Activation('relu')(x)
    x4 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)

    x5 = concatenate([x3, x4], axis=3)
    x = BatchNormalization()(x5)
    x = Activation('relu')(x)
    x6 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)

    x7 = concatenate([x5, x6], axis=3)
    x = BatchNormalization()(x7)
    x = Activation('relu')(x)
    x8 = Conv2D(124, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)

    x = BatchNormalization()(x8)
    x = Activation('relu')(x)
    x9 = Conv2D(124, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
    x9 = MaxPooling2D(pool_size=(2, 2))(x9)
    return x9



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

    inputs = Input(shape=(28, 28, 1))

    x = densenet(inputs)
    x = densenet(x)
    x = densenet(x)

    # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
    x = Flatten()(x)

    x = Dense(256, activation='relu')(x)
    x = Dense(10, activation='sigmoid')(x)

    # 确定模型
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])#编译模型
    model.fit(x_train, y_train, nb_epoch=10, batch_size=64, validation_data=(x_test, y_test), shuffle=True)#训练模型
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    yaml_string = model.to_yaml()
    with open('./models/DenseNet.yaml', 'w') as outfile:
        outfile.write(yaml_string)
    model.save_weights('./models/DenseNet.h5')


