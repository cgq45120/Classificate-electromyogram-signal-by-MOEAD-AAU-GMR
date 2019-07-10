from dealxh1024gj import *
import numpy as np
import random
import time
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import svm

class Moead_AAU(object):
    def __init__(self):
        self.partern = 15  # 相邻种群
        self.iteration = 1  # 迭代次数
        self.passageway = 16  # 通道数目
        self.feature = 14  # 特征数目
        self.feature_low = 4
        self.passageway_low = 6
        self.popSize = self.passageway + self.feature  # 子链纬度
        self.obj = 4  # 目标数目
        self.F = 1  # 进化的F
        self.cr = 0.5  # 交叉概率
        self.row = 11  # 生成纬度
        self.loaddata()

    def loaddata(self):
        model_deal = dealxh()
        self.trainDate, self.testDate = model_deal.readFile()
        self.trainRow = self.trainDate.shape[0]
        self.testRow = self.testDate.shape[0]
        trainTag = []
        testTag = []
        for i in range(5):  # 动作为5个
            for j in range(int(self.trainRow/5)):  # 标签为390个
                trainTag.append(i+1)
            for j in range(int(self.testRow/5)):
                testTag.append(i+1)
        self.trainTag = np.array(trainTag)
        self.testTag = np.array(testTag)
        print('import data')
        print(time.ctime())

    def getpartern(self):  # 生成邻居矩阵，通过计算权重之间的欧式距离来生成最近15个邻居位置
        f = []
        for i in range(self.row):
            for j in range(self.row-i):
                for k in range(self.row-i-j):
                    s = []
                    s.append(i)
                    s.append(j)
                    s.append(k)
                    t = self.row - 1 - i - j - k
                    s.append(t)
                    f.append(s)
        self.weight = np.array(f) / 10+0.000001
        self.sonSize = np.size(self.weight, 0)
        distance = np.zeros((self.sonSize, self.sonSize))
        for i in range(self.sonSize):
            distance[i, :] = np.linalg.norm(
                (self.weight - np.tile(self.weight[i, :], (self.sonSize, 1))), axis=1)
        distanceIndex = np.zeros((self.sonSize, self.partern))
        for i in range(self.sonSize):
            test = distance[i, :]
            tag = [i for i in range(self.sonSize)]
            test = np.c_[test, tag]  # 将标签加上去
            test = np.transpose(test)
            for j in range(self.sonSize-1):  # 冒泡排序选取最小的20个距离的标签
                for k in range(self.sonSize-j-1):
                    if test[0, k] > test[0, k+1]:
                        test[0, k], test[0, k+1] = test[0, k+1], test[0, k]
                        test[1, k], test[1, k+1] = test[1, k+1], test[1, k]
            distanceIndex[i, :] = test[1, 0:15]
            self.distanceIndex = distanceIndex.astype(np.int)

    def getson(self):  # 生成不同权重下的种群
        popSon = np.random.randint(0, 2, size=(self.sonSize, self.popSize))
        for i in range(self.sonSize):
            while sum(popSon[i, 0:self.feature]) < self.feature_low or sum(popSon[i, self.feature:self.popSize]) < self.passageway_low:  # 规定通道数>=4并且通道数>=6
                popSon[i, :] = [random.randint(0, 1)
                                for j in range(self.popSize)]
        return popSon

    def calculatorZ(self, popSon):  # 计算fitness以及最小值
        y = np.zeros((self.sonSize, self.obj))
        for i in range(self.sonSize):
            y[i, :] = self.fitness(popSon[i, :])
        zmin = np.min(y, axis=0)
        return y, zmin

    def geneticmoead(self, y, zmin, popSon):
        weight = 1/self.weight
        Gmax = 5
        normailzation = np.array([self.feature-self.feature_low,self.passageway-self.passageway_low,1,1])
        for i in range(self.iteration):  # 选取3个邻居进行差分进化，使用切比雪夫分解法得到帕累托最优解
            # update_time = 0
            if i % 10 == 0:
                print(i)
            G = int(Gmax / (1 + math.exp(-20 * ((i + 1) / self.iteration - 0.5)))) + 1
            s = [0, 0, 0]
            t = np.zeros((self.sonSize, self.popSize))
            while s[0] == s[1] or s[0] == s[2] or s[1] == s[2]:  # 判断随机3个邻居是否重叠
                s = [random.randint(0, self.partern - 1) for k in range(3)]
            crossrate = [random.random() for k in range(self.popSize)]
            index = [random.randint(0, self.popSize-1) for i in range(self.sonSize)]  # 必定变异位置
            t = popSon[self.distanceIndex[:, s[0]], :] + self.F * (popSon[self.distanceIndex[:, s[1]], :] - popSon[self.distanceIndex[:, s[2]], :])  # 差分进化
            for j in range(self.sonSize):
                for k in range(self.popSize):
                    if crossrate[k] > self.cr and k != index[j]:
                        t[j, k] = popSon[j, k]
            t_regular = t.copy()
            t_regular[t < 0.5] = 0
            t_regular[t >= 0.5] = 1
            t = t_regular
            for j in range(self.sonSize):
                if sum(t[j, 0:self.feature]) >= self.feature_low and sum(t[j, self.feature:self.popSize]) >= self.passageway_low:
                    ynow = self.fitness(t[j, :])
                    angle = np.dot(weight, (ynow - zmin).T) / (np.linalg.norm(weight, axis=1) * np.linalg.norm((ynow - zmin + 1e-8)))
                    tag = [k for k in range(self.sonSize)]
                    angletest = np.c_[angle, tag]  # 将标签加上去
                    angletest = np.transpose(angletest)
                    for k in range(self.sonSize - 1):  
                        for l in range(self.sonSize - j - 1):
                            if angletest[0, l] < angletest[0, l + 1]:
                                angletest[0, l], angletest[0, l +
                                                        1] = angletest[0, l + 1], angletest[0, l]
                                angletest[1, l], angletest[1, l +
                                                        1] = angletest[1, l + 1], angletest[1, l]
                    angleIndex = angletest[1, 0:G]
                    for l in range(G):
                        if np.dot(weight[int(angleIndex[l]), :], ((ynow - zmin)/normailzation).T) < np.dot(
                                weight[int(angleIndex[l]), :], ((y[int(angleIndex[l]), :] - zmin)/normailzation).T):
                            popSon[int(angleIndex[l]), :] = t[j, :]
                            y[int(angleIndex[l]), :] = ynow
                            # update_time+=1
            ynow_min = y.min(0)
            if ynow_min[0] < zmin[0]:  # 更新最小值
                zmin[0] = ynow_min[0]
            if ynow_min[1] < zmin[1]:
                zmin[1] = ynow_min[1]
            if ynow_min[2] < zmin[2]:
                zmin[2] = ynow_min[2]
            if ynow_min[3] < zmin[3]:
                zmin[3] = ynow_min[3]
            # print(update_time)
        # out_result = np.sum(popSon, axis=0) / popSon.shape[0]
        # out_result_numb = np.unique(popSon, axis=0).shape[0]
        with open('result_moeadau'+str(self.feature_low)+'_'+str(self.passageway_low)+'.txt', 'w') as f:
            for i in range(self.sonSize):
                f.write('特征数:'+str(y[i, 0])+' 通道数:'+str(y[i, 1]) +
                        ' 准确率:'+str(1-y[i, 2])+' 准确率标准差:'+str(y[i, 3]))
                f.write('\n')
                f.write(str(popSon[i, :]))
                f.write('\n')
            # f.write(str(out_result))
            # f.write('\n')
            # f.write(str(out_result_numb))
        self.ploturf(y)

    def ploturf(self, y):
        xaxis = np.unique(y[:, 0])
        yaxis = np.unique(y[:, 1])
        xlength = xaxis.shape[0]
        ylength = yaxis.shape[0]
        plotfinal1 = []
        plotfinal2 = []
        for i in range(xlength):
            for j in range(ylength):
                f = []
                for k in range(self.sonSize):
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
        plotfinal1 = np.array(plotfinal1)
        plotfinal2 = np.array(plotfinal2)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(plotfinal1[:, 0], plotfinal1[:, 1],
                        plotfinal1[:, 2], color='b')
        ax.plot_trisurf(plotfinal2[:, 0], plotfinal2[:, 1],
                        plotfinal2[:, 2], color='b')
        ax.set_xlim(self.feature_low, 14)
        ax.set_ylim(self.passageway_low, 16)
        ax.set_xlabel('feature')
        ax.set_ylabel('channal')
        ax.set_zlabel('accuracy')
        plt.show()
        plt.savefig('moead_au'+str(self.feature_low)+'_' + str(self.passageway_low)+'.png', dpi=500)

    def fitness(self, popSon):  # 计算适应度，返回的分别是特征数、通道数、准确率、方差
        y = []
        y.append(sum(popSon[0:self.feature]))
        y.append(sum(popSon[self.feature:self.passageway + self.feature]))
        popMatix = np.dot(np.matrix(popSon[0:self.feature]).T, np.matrix(
            popSon[self.feature:self.passageway + self.feature]))
        avg, std = self.svmclassify(popMatix)
        avg = 1-avg
        y.append(avg)
        y.append(std)
        return y

    def svmclassify(self, popMatix):  # 对信息进行选择后使用SVM进行分类
        popMatix = popMatix.reshape((1, 224))  # 用0-1特征通道矩阵对信息进行选择
        popMatix1 = np.tile(popMatix, (self.trainRow, 1))
        popMatix2 = np.tile(popMatix, (self.testRow, 1))
        trainDate = np.multiply(popMatix1, self.trainDate)
        testDate = np.multiply(popMatix2, self.testDate)
        clf = svm.SVC(decision_function_shape='ovo', gamma='auto')
        clf.fit(trainDate, self.trainTag)
        variousTag1 = np.unique(self.trainTag)
        rowacc = variousTag1.shape[0]   # rowacc表示动作数目
        accNumb = np.zeros((1, rowacc))
        for i in range(self.testRow):  # 判断测试集的分类
            k = np.matrix(testDate[i, :])
            y = clf.predict(k)
            if y == self.testTag[i]:
                accNumb[0, y-1] += 1
        accNumb = accNumb/195
        avg = sum(sum(accNumb) / rowacc)
        std = math.sqrt(sum(sum((accNumb - avg) ** 2))/5)
        return avg, std

    def main(self):
        self.getpartern()  # 生成权重向量
        popSon = self.getson()
        y, zmin = self.calculatorZ(popSon)
        print('begin train')
        print(time.ctime())
        self.geneticmoead(y, zmin, popSon)
        print(time.ctime())


if __name__ == '__main__':
    print(time.ctime())
    model = Moead_AAU()
    model.main()






