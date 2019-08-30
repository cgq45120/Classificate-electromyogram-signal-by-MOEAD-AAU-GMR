from deal_feature import *
import numpy
import random
import time
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import svm


def getpartern(partern, row):  # 生成邻居矩阵，通过计算权重之间的欧式距离来生成最近15个邻居位置
    f = []
    for i in range(row):
        for j in range(row-i):
            for k in range(row-i-j):
                s = []
                s.append(i)
                s.append(j)
                s.append(k)
                t = row - 1 - i - j - k
                s.append(t)
                f.append(s)
    weight = numpy.array(f) / 10+0.000001
    sonSize = numpy.size(weight, 0)
    distance = numpy.zeros((sonSize, sonSize))
    for i in range(sonSize):
        distance[i, :] = numpy.linalg.norm(
            (weight - numpy.tile(weight[i, :], (sonSize, 1))), axis=1)
    distanceIndex = numpy.zeros((sonSize, partern))
    for i in range(sonSize):
        test = distance[i, :]
        tag = [i for i in range(sonSize)]
        test = numpy.c_[test, tag]  # 将标签加上去
        test = numpy.transpose(test)
        for j in range(sonSize-1):  # 冒泡排序选取最小的20个距离的标签
            for k in range(sonSize-j-1):
                if test[0, k] > test[0, k+1]:
                    test[0, k], test[0, k+1] = test[0, k+1], test[0, k]
                    test[1, k], test[1, k+1] = test[1, k+1], test[1, k]
        distanceIndex[i, :] = test[1, 0:15]
        distanceIndex = distanceIndex.astype(numpy.int)
    return distanceIndex, weight, sonSize  # 返回邻居矩阵，权重以及种群要生成的数目


def getson(sonSize, popSize, feature, feature_low, passageway_low):  # 生成不同权重下的种群
    popSon = numpy.random.randint(0, 2, size=(sonSize, popSize))
    for i in range(sonSize):
        while sum(popSon[i, 0:feature]) < feature_low or sum(popSon[i, feature:popSize]) < passageway_low:  # 规定通道数>=4并且通道数>=6
            popSon[i, :] = [random.randint(0, 1) for j in range(popSize)]
    return popSon  # 返回种群


def calculatorZ(popSon, sonSize, obj, trainDate, trainTag, testDate, testTag, feature, passageway, trainRow, testRow):  # 计算fitness以及最小值
    y = numpy.zeros((sonSize, obj))
    for i in range(sonSize):
        y[i, :] = fitness(popSon[i, :], trainDate, trainTag, testDate,
                          testTag, feature, passageway, trainRow, testRow)
    zmin = numpy.min(y, axis=0)
    return y, zmin  # 返回y值以及理想点Z


def geneticmoead(y, weight, zmin, popSon, sonSize, popSize, partern, distanceIndex, iteration, F, cr, trainDate, trainTag, testDate, testTag, feature, passageway, trainRow, testRow, feature_low, passageway_low):
    weight = 1/weight
    Gmax = 5
    normailzation = numpy.array([feature-feature_low,passageway-passageway_low,1,1])
    for i in range(iteration):  # 选取3个邻居进行差分进化，使用切比雪夫分解法得到帕累托最优解
        update_time = 0
        if i % 10 == 0:
            print(i)
        G = int(Gmax / (1 + math.exp(-20 * ((i + 1) / iteration - 0.5)))) + 1
        s = [0, 0, 0]
        t = numpy.zeros((sonSize, popSize))
        while s[0] == s[1] or s[0] == s[2] or s[1] == s[2]:  # 判断随机3个邻居是否重叠
            s = [random.randint(0, partern - 1) for k in range(3)]
        crossrate = [random.random() for k in range(popSize)]
        index = [random.randint(0, popSize-1)
                 for i in range(sonSize)]  # 必定变异位置
        t = popSon[distanceIndex[:, s[0]], :] + F * \
            (popSon[distanceIndex[:, s[1]], :] -
             popSon[distanceIndex[:, s[2]], :])  # 差分进化
        for j in range(sonSize):
            for k in range(popSize):
                if crossrate[k] > cr and k != index[j]:
                    t[j, k] = popSon[j, k]
        t_regular = t.copy()
        t_regular[t < 0.5] = 0
        t_regular[t >= 0.5] = 1
        t = t_regular
        for j in range(sonSize):
            if sum(t[j, 0:feature]) >= feature_low and sum(t[j, feature:popSize]) >= passageway_low:
                ynow = fitness(t[j, :], trainDate, trainTag, testDate,
                               testTag, feature, passageway, trainRow, testRow)
                angle = numpy.dot(weight, (ynow - zmin).T) / (
                    numpy.linalg.norm(weight, axis=1) * numpy.linalg.norm((ynow - zmin + 1e-8)))
                tag = [k for k in range(sonSize)]
                angletest = numpy.c_[angle, tag]  # 将标签加上去
                angletest = numpy.transpose(angletest)
                for k in range(sonSize - 1):  
                    for l in range(sonSize - j - 1):
                        if angletest[0, l] < angletest[0, l + 1]:
                            angletest[0, l], angletest[0, l +
                                                       1] = angletest[0, l + 1], angletest[0, l]
                            angletest[1, l], angletest[1, l +
                                                       1] = angletest[1, l + 1], angletest[1, l]
                angleIndex = angletest[1, 0:G]
                for l in range(G):
                    if numpy.dot(weight[int(angleIndex[l]), :], ((ynow - zmin)/normailzation).T) < numpy.dot(
                            weight[int(angleIndex[l]), :], ((y[int(angleIndex[l]), :] - zmin)/normailzation).T):
                        popSon[int(angleIndex[l]), :] = t[j, :]
                        y[int(angleIndex[l]), :] = ynow
                        update_time+=1
        ynow_min = y.min(0)
        if ynow_min[0] < zmin[0]:  # 更新最小值
            zmin[0] = ynow_min[0]
        if ynow_min[1] < zmin[1]:
            zmin[1] = ynow_min[1]
        if ynow_min[2] < zmin[2]:
            zmin[2] = ynow_min[2]
        if ynow_min[3] < zmin[3]:
            zmin[3] = ynow_min[3]
        print(update_time)
    out_result = numpy.sum(popSon, axis=0)/popSon.shape[0]
    out_result_numb = numpy.unique(popSon, axis=0).shape[0]
    with open('result_moeadau'+str(feature_low)+'_'+str(passageway_low)+'.txt', 'w') as f:
        for i in range(sonSize):
            f.write('特征数:'+str(y[i, 0])+' 通道数:'+str(y[i, 1]) +
                    ' 准确率:'+str(1-y[i, 2])+' 准确率标准差:'+str(y[i, 3]))
            f.write('\n')
            f.write(str(popSon[i, :]))
            f.write('\n')
        f.write(str(out_result))
        f.write('\n')
        f.write(str(out_result_numb))
    ploturf(y, sonSize, feature_low, passageway_low)


def ploturf(y, sonSize, feature_low, passageway_low):
    xaxis = numpy.unique(y[:, 0])
    yaxis = numpy.unique(y[:, 1])
    xlength = xaxis.shape[0]
    ylength = yaxis.shape[0]
    plotfinal1 = []
    plotfinal2 = []
    for i in range(xlength):
        for j in range(ylength):
            f = []
            for k in range(sonSize):
                if y[k, 0] == xaxis[i] and y[k, 1] == yaxis[j]:
                    f.append(1-y[k, 2])
            if len(f):
                final = []
                final.append(xaxis[i])
                final.append(yaxis[j])
                final.append(min(f))
                plotfinal1.append(final)
                final = []
                final.append(xaxis[i])
                final.append(yaxis[j])
                final.append(max(f))
                plotfinal2.append(final)
    plotfinal1 = numpy.array(plotfinal1)
    plotfinal2 = numpy.array(plotfinal2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(plotfinal1[:, 0], plotfinal1[:, 1],
                    plotfinal1[:, 2], color='b')
    ax.plot_trisurf(plotfinal2[:, 0], plotfinal2[:, 1],
                    plotfinal2[:, 2], color='b')
    ax.set_xlim(feature_low, 14)
    ax.set_ylim(passageway_low, 16)
    ax.set_xlabel('feature')
    ax.set_ylabel('channal')
    ax.set_zlabel('accuracy')
    plt.show()
    plt.savefig('moead_au'+str(feature_low)+'_' +
                str(passageway_low)+'.png', dpi=500)


def fitness(x, trainDate, trainTag, testDate, testTag, feature, passageway, trainRow, testRow):  # 计算适应度，返回的分别是特征数、通道数、准确率、方差
    y = []
    y.append(sum(x[0:feature]))
    y.append(sum(x[feature:passageway+feature]))
    popMatix = numpy.dot(numpy.matrix(x[0:feature]).T, numpy.matrix(
        x[feature:passageway+feature]))
    avg, std = svmclassify(popMatix, trainDate, trainTag,
                           testDate, testTag, trainRow, testRow)
    avg = 1-avg
    y.append(avg)
    y.append(std)
    return y


def svmclassify(popMatix, trainDate, trainTag, testDate, testTag, trainRow, testRow):  # 对信息进行选择后使用SVM进行分类
    popMatix = popMatix.reshape((1, 224))  # 用0-1特征通道矩阵对信息进行选择
    popMatix1 = numpy.tile(popMatix, (trainRow, 1))
    popMatix2 = numpy.tile(popMatix, (testRow, 1))
    trainDate = numpy.multiply(popMatix1, trainDate)
    testDate = numpy.multiply(popMatix2, testDate)
    clf = svm.SVC(decision_function_shape='ovo', gamma='auto')
    clf.fit(trainDate, trainTag)
    variousTag1 = numpy.unique(trainTag)
    rowacc = variousTag1.shape[0]   # rowacc表示动作数目
    accNumb = numpy.zeros((1, rowacc))
    for i in range(testRow):  # 判断测试集的分类
        k = numpy.matrix(testDate[i, :])
        y = clf.predict(k)
        if y == testTag[i]:
            accNumb[0, y-1] += 1
    accNumb = accNumb/195
    avg = sum(sum(accNumb) / rowacc)
    std = math.sqrt(sum(sum((accNumb - avg) ** 2))/5)
    return avg, std


if __name__ == '__main__':
    print(time.ctime())
    partern = 15  # 相邻种群
    iteration = 100  # 迭代次数
    passageway = 16  # 通道数目
    feature = 14  # 特征数目
    feature_low = 4
    passageway_low = 6
    popSize = passageway + feature  # 子链纬度
    obj = 4  # 目标数目
    F = 1  # 进化的F
    cr = 0.5  # 交叉概率
    row1 = 11  # 生成纬度
    b = dealxh()
    trainDate, testDate = b.readFile()
    trainRow = trainDate.shape[0]
    testRow = testDate.shape[0]
    trainTag = []
    testTag = []
    for i in range(5):  # 动作为5个
        tag1 = [i+1 for j in range(int(trainRow/5))]  # 标签为390个
        trainTag = trainTag + tag1
        tag2 = [i+1 for j in range(int(testRow/5))]
        testTag = testTag + tag2
    trainTag = numpy.array(trainTag)
    testTag = numpy.array(testTag)
    print('import data')
    print(time.ctime())
    distanceIndex, weight, sonSize = getpartern(partern, row1)
    popSon = getson(sonSize, popSize, feature, feature_low, passageway_low)
    y, zmin = calculatorZ(popSon, sonSize, obj, trainDate, trainTag,
                          testDate, testTag, feature, passageway, trainRow, testRow)
    print('begin train')
    print(time.ctime())
    geneticmoead(y, weight, zmin, popSon, sonSize, popSize, partern, distanceIndex, iteration, F, cr, trainDate,
                 trainTag, testDate, testTag, feature, passageway, trainRow, testRow, feature_low, passageway_low)
