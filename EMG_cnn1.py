# coding=utf-8

# 使用自己数据,输入300*16矩阵数据,加传统分类器

import tensorflow as tf
import numpy as np
import os
import scipy.io as sio
import string
import math
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

path1 = 'F:/CCC/人工智能/moead/all_data_wrist_train/'
path2 = 'F:/CCC/人工智能/moead/all_data_wrist_test/'

right1 = 0.0
right2 = 0.0
right3 = 0.0

# 读取肌电数据
def LoadData(p1,p2):
    X_train_orig = np.zeros((40950,300,16,1))   #44850
    X_test_orig = np.zeros((1,40950),dtype=int)
    Y_train_orig = np.zeros((5850,300,16,1))   #5850
    Y_test_orig = np.zeros((1,5850),dtype=int)

    m = 0
    for files1 in os.listdir(p1):
        EMG = sio.loadmat(p1 + files1)
        data = EMG['data']
        for i in range(195):
            X_train_orig[m * 195 + i, :, :, 0] = data[i*50:i*50+300, :]
            X_test_orig[0, m * 195 + i] = int(files1[6:7])
        m = m + 1

    n = 0
    for files2 in os.listdir(p2):
        EMG = sio.loadmat(p2 + files2)
        data = EMG['data']
        for i in range(195):
            Y_train_orig[n * 195 + i, :, :, 0] = data[i*50:i*50+300, :]
            Y_test_orig[0, n * 195 + i] = int(files2[6:7])
        n = n + 1

    return X_train_orig,X_test_orig,Y_train_orig,Y_test_orig

# 转换label成矩阵
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

'''
def weight_variable(shape):
    # 用正态分布来初始化权值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 本例中用relu激活函数，所以用一个很小的正偏置较好
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def batch_normalization(bnin):
    batch_mean,batch_var=tf.nn.moments(bnin,[0,1,2],keep_dims=True)
    shift=tf.Variable(tf.zeros([batch_mean]))
    scale=tf.Variable(tf.ones([batch_mean]))
    BN_out=tf.nn.batch_normalization(bnin,batch_mean,batch_var,shift,scale,epsilon)
    return BN_out
'''

def create_placeholders(n_H0,n_W0,n_C0,n_y):
    X=tf.placeholder('float',[None,n_H0,n_W0,n_C0])
    Y=tf.placeholder('float',[None,n_y])
    return X,Y

# 提取随机的batch进行训练
def random_mini_batches(X, Y, mini_batch_size=10):

    m = X.shape[0]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m / mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

# 后来发现本身读入的时候就是随机读入的，其实这里感觉也不一定需要再进行随机提取
def mini_batches(X, Y, mini_batch_size=10):

    m = X.shape[0]  # number of training examples
    batches = []

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m / mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        batches.append(mini_batch)

    return batches

# 对数据做一个归一化处理，可以提速
def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()

def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape


g = tf.Graph()
with g.as_default():
    X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = LoadData(path1,path2)
    X_train = X_train_orig
    Y_train = Y_train_orig
    X_test = convert_to_one_hot(X_test_orig,10).T
    Y_test = convert_to_one_hot(Y_test_orig,10).T

# variable learning rate
    lr = tf.placeholder(tf.float32)
# test flag for batch norm
    tst = tf.placeholder(tf.bool)
    iter = tf.placeholder(tf.int32)
# dropout probability
    pkeep = tf.placeholder(tf.float32)
    pkeep_conv = tf.placeholder(tf.float32)

    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = X_test.shape[1]
    X, Y_ = create_placeholders(n_H0, n_W0, n_C0, n_y)

#1:卷积
    B0 = tf.Variable(tf.constant(0.1, tf.float32, [1]))
    X1, update_ema0 = batchnorm(X, tst, iter, B0)
    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 64], stddev=0.1))
    B1 = tf.Variable(tf.constant(0.1, tf.float32, [64]))
    Y1l = tf.nn.conv2d(X1, W1, strides=[1, 1, 1, 1], padding='SAME')
    Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1, convolutional=True)
    Y11 = tf.nn.relu(Y1bn)

    Y1 = tf.nn.max_pool(Y11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#2:卷积
    W2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    B2 = tf.Variable(tf.constant(0.1, tf.float32, [64]))
    Y2l = tf.nn.conv2d(Y1, W2, strides=[1, 1, 1, 1], padding='SAME')
    Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2, convolutional=True)
    Y22 = tf.nn.relu(Y2bn)

    Y2 = tf.nn.max_pool(Y22, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#3:本地连接层
    W3 = tf.Variable(tf.truncated_normal([1, 1, 64, 64], stddev=0.1))
    B3 = tf.Variable(tf.constant(0.1, tf.float32, [64]))
    Y3l = tf.nn.conv2d(Y2, W3, strides=[1, 1, 1, 1], padding='SAME')
    Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3, convolutional=True)
    Y3 = tf.nn.relu(Y3bn)

#4:本地连接层
    W4 = tf.Variable(tf.truncated_normal([1, 1, 64, 64], stddev=0.1))
    B4 = tf.Variable(tf.constant(0.1, tf.float32, [64]))
    Y4l = tf.nn.conv2d(Y3, W4, strides=[1, 1, 1, 1], padding='SAME')
    Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4, convolutional=True)
    Y4r = tf.nn.relu(Y4bn)
    Y4 = tf.nn.dropout(Y4r, pkeep_conv, compatible_convolutional_noise_shape(Y4r))

# reshape the output from the third convolution for the fully connected layer
    YY = tf.reshape(Y4, shape=[-1, 64*75*4])

#5:全连接层
    W5 = tf.Variable(tf.truncated_normal([64*75*4, 512], stddev=0.1))
    B5 = tf.Variable(tf.constant(0.1, tf.float32, [512]))
    Y5l = tf.matmul(YY, W5)
    Y5bn, update_ema5 = batchnorm(Y5l, tst, iter, B5)
    Y5r = tf.nn.relu(Y5bn)
    Y5 = tf.nn.dropout(Y5r, pkeep)

#6:全连接层
    W6 = tf.Variable(tf.truncated_normal([512, 256], stddev=0.1))
    B6 = tf.Variable(tf.constant(0.1, tf.float32, [256]))
    Y6l = tf.matmul(Y5, W6)
    Y6bn, update_ema6 = batchnorm(Y6l, tst, iter, B6)
    Y6r = tf.nn.relu(Y6bn)
    Y6 = tf.nn.dropout(Y6r, pkeep)

#7:全连接层
    W7 = tf.Variable(tf.truncated_normal([256, 128], stddev=0.1))
    B7 = tf.Variable(tf.constant(0.1, tf.float32, [128]))
    Y7l = tf.matmul(Y6, W7)
    Y7bn, update_ema7 = batchnorm(Y7l, tst, iter, B7)
    Y7 = tf.nn.relu(Y7bn)

#8:全连接层
    W8 = tf.Variable(tf.truncated_normal([128, 64], stddev=0.1))
    B8 = tf.Variable(tf.constant(0.1, tf.float32, [64]))
    Y8l = tf.matmul(Y7, W8)
    Y8bn, update_ema8 = batchnorm(Y8l, tst, iter, B8)
    Y8 = tf.nn.relu(Y8bn)

#9:全连接层
    W9 = tf.Variable(tf.truncated_normal([64, 32], stddev=0.1))
    B9 = tf.Variable(tf.constant(0.1, tf.float32, [32]))
    Y9l = tf.matmul(Y8, W9)
    Y9bn, update_ema9 = batchnorm(Y9l, tst, iter, B9)
    Y9 = tf.nn.relu(Y9bn)

#10:全连接层
    W10 = tf.Variable(tf.truncated_normal([32, 16], stddev=0.1))
    B10 = tf.Variable(tf.constant(0.1, tf.float32, [16]))
    Y10l = tf.matmul(Y9, W10)
    Y10bn, update_ema10 = batchnorm(Y10l, tst, iter, B10)
    Y10 = tf.nn.relu(Y10bn)

#标记，方便后面保存网络特征
    getfeature = tf.reshape(Y10, [-1, 16], name='getfeature')

#11:输出
    W11 = tf.Variable(tf.truncated_normal([16, 10], stddev=0.1))
    B11 = tf.Variable(tf.constant(0.1, tf.float32, [10]))
    Ylogits = tf.matmul(Y10, W11) + B11

    Y = tf.nn.softmax(Ylogits)

    update_ema = tf.group(update_ema0, update_ema1, update_ema2, update_ema3, update_ema4, update_ema5, update_ema6, update_ema7, update_ema8, update_ema9, update_ema10)

# 损失函数：cross_entropy
#    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,labels=Y_))
#    regularization_loss = tf.reduce_mean(tf.square(W11))
#    hinge_loss = tf.reduce_mean(tf.square(tf.maximum(tf.zeros([25, 10]), 1 - Y_ * Ylogits)))
#    cross_entropy = regularization_loss + 0.9 * hinge_loss  #  penalty_parameter = 1

    cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# 预测准确结果统计
# 预测值中最大值（１）即分类结果，是否等于原始标签中的（１）的位置。argmax()取最大值所在的下标
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#accuracy_1 = tf.argmax(tf.bincount(tf.cast(correct_prediction, tf.float32)))

# 优化函数：AdamOptimizer
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

'''
# 如果一次性来做测试的话，可能占用的显存会比较多，所以测试的时候也可以设置较小的batch来看准确率
test_acc_sum = tf.Variable(0.0)
batch_acc = tf.placeholder(tf.float32)
new_test_acc_sum = tf.add(test_acc_sum, batch_acc)
update = tf.assign(test_acc_sum, new_test_acc_sum)
'''

with tf.Session(graph=g) as sess:
    # Run the initialization
    sess.run(tf.global_variables_initializer())

# 训练
    for i in range(35):
        j=0
        acc=0
        batchs = random_mini_batches(X_train, X_test, 25)
        for batch in batchs:
            j+=1
        # Select a minibatch
            (batch_X, batch_Y) = batch
#　　　　　　　learning rate decay
#            max_learning_rate = 0.02
#            min_learning_rate = 0.0001
#            decay_speed = 2000
#            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-j / decay_speed)
            learning_rate = 0.0001

        # compute training values for visualisation
            if (i+1) % 1 == 0:
                a, c = sess.run([accuracy, cross_entropy],{X: batch_X, Y_: batch_Y, tst: False, pkeep: 0.5, pkeep_conv: 0.5})
                acc += a
                if j == 1170:   # 40950/50
                    acc=acc/j
                    print("training step: " + str(i+1) + " loss: " + str(c) + " acc: " + str(acc))

        # the backpropagation training step
            sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False, pkeep: 0.5, pkeep_conv: 0.5})
            sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 0.5, pkeep_conv: 0.5})

# 测试
    step = 0
    num = 0
    batchs = mini_batches(Y_train, Y_test, 25)
    for batch in batchs:
        (batch_X, batch_Y) = batch

        step += 1
        test_acc = sess.run(accuracy, feed_dict={X: batch_X, Y_: batch_Y, tst: True, pkeep: 0.5, pkeep_conv: 0.5})

        num += test_acc
        if step % 5 == 0:
            print("testing step %d, test_acc %g" % (step, test_acc))

    print("test accuracy: " + str(num / step))