import numpy as np
import math
import time
class dealxh(object):
    def __init__(self):
        self.feature = 14
        self.channal = 16

    def sumRMS(self,signalOrigin):
        sumRms = np.sqrt(np.sum(signalOrigin**2/300,axis=0))
        return sumRms

    def sumMAV(self,signalOrigin):
        sumMAV = np.sum(abs(signalOrigin),axis=0)
        return sumMAV

    def sumWL(self,signalOrigin):
        sumWL = np.sum(abs(signalOrigin[0:299,:] - signalOrigin[1:,:])/299,axis=0)
        return sumWL

    def sumZC(self,signalOrigin):
        condition1 = (abs(signalOrigin[0:299,:]-signalOrigin[1:,:])>50)+0
        condition2 = (np.multiply(signalOrigin[0:299,:],signalOrigin[1:,:])<0)+0
        sumZC = np.sum(((condition1+condition2)>1)+0,axis=0)
        return sumZC

    def sumDASDV(self,signalOrigin):
        sumDASDV = np.sqrt(np.sum((signalOrigin[0:299,:] - signalOrigin[1:,:])**2/299,axis=0))
        return sumDASDV

    def sumLOG(self,signalOrigin):
        sumLOG = np.exp(np.sum(signalOrigin/300,axis=0))
        return sumLOG

    def sumSII(self,signalOrigin):
        sumSII = np.sum(signalOrigin**2,axis=0)
        return sumSII

    def sumTM3(self,signalOrigin):
        sumTM3 = abs(np.sum(signalOrigin**3/300,axis=0))
        return sumTM3

    def sumTM4(self,signalOrigin):
        sumTM4 = abs(np.sum(signalOrigin**4/300,axis=0))
        return sumTM4

    def sumTM5(self,signalOrigin):
        sumTM5 = abs(np.sum(signalOrigin**5/300,axis=0))
        return sumTM5

    def frequencyRatio(self,signalOrigin):
        sign = abs(np.fft.fft(signalOrigin, 512,axis = 0) / 300)
        sumfrequencyRatio = np.min(sign, axis = 0)/np.max(sign, axis = 0)
        return sumfrequencyRatio

    def sumIEMG(self,signalOrigin):
        sumIEMG = np.sum(abs(signalOrigin)/300,axis=0)
        return sumIEMG

    def sumMFMN(self,signalOrigin):
        f = 1000*np.linspace(0,1,512)
        sign = abs(np.fft.fft(signalOrigin, 512,axis = 0) / 300)
        sumMFMN = np.sum(np.multiply(np.tile(f,(16,1)).T,sign),axis=0)/np.sum(sign,axis=0)
        return sumMFMN

    def sumMFMD(self,signalOrigin):
        sign = abs(np.fft.fft(signalOrigin, 512,axis = 0) / 300)
        sumMFMD = np.sum(sign, axis = 0)/2
        return sumMFMD

    def readFile(self):
        trainOrigin = np.loadtxt('./data/muscle_train.txt')
        testOrigin = np.loadtxt('./data/muscle_test.txt')
        trainData = self.deal(trainOrigin)
        testData = self.deal(testOrigin)
        trainData,testData = self.onehot(trainData,testData)
        return trainData,testData


    def onehot(self,trainData,testData):
        one_hot = np.vstack((trainData, testData))
        one_hot = (one_hot - one_hot.min(0))/(one_hot.max(0) - one_hot.min(0))
        trainData = one_hot[0:trainData.shape[0], :]
        testData = one_hot[trainData.shape[0]:, :]
        return trainData,testData
        
    def deal(self,trainOrigin):
        trainRow = trainOrigin.shape[0]
        Data = []
        passID = []
        for i in range(int(trainRow/10000)):
            passID.append(i*200+195)
            passID.append(i*200+196)
            passID.append(i*200+197)
            passID.append(i*200+198)
            passID.append(i*200+199)
        for i in range(int(trainRow/50)):
            if i in passID:
                continue
            else:
                j = i*50+300
                RMS = self.sumRMS(trainOrigin[i*50:j,:])
                MAV = self.sumMAV(trainOrigin[i*50:j,:])
                WL = self.sumWL(trainOrigin[i*50:j,:])
                ZC = self.sumZC(trainOrigin[i*50:j,:])
                DASDV = self.sumDASDV(trainOrigin[i*50:j,:])
                LOG = self.sumLOG(trainOrigin[i*50:j,:])
                SII = self.sumSII(trainOrigin[i*50:j,:])
                TM3 = self.sumTM3(trainOrigin[i*50:j,:])
                TM4 = self.sumTM4(trainOrigin[i*50:j,:])
                TM5 = self.sumTM5(trainOrigin[i*50:j,:])
                frequencyRatio = self.frequencyRatio(trainOrigin[i*50:j,:])
                MFMN = self.sumMFMN(trainOrigin[i*50:j,:])
                MFMD = self.sumMFMD(trainOrigin[i*50:j,:])
                IEMG = self.sumIEMG(trainOrigin[i*50:j,:])
                Data_feature = [RMS,MAV,WL,ZC,DASDV,LOG,SII,TM3,TM4,TM5,frequencyRatio,MFMN,MFMD,IEMG]
                Data.append(Data_feature)
        Data = np.array(Data).reshape((-1,self.feature*self.channal))
        return Data
if __name__ == "__main__":
    A= dealxh()
    A.readFile()