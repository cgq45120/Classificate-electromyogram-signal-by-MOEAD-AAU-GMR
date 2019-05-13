import numpy
import math
import time
class dealxh(object):
    def sumRMS(self,signalOrigin):
        sumRms = numpy.sqrt(numpy.sum(signalOrigin**2/300,axis=0))
        sumRms = sumRms.tolist()
        return sumRms
    def sumMAV(self,signalOrigin):
        sumMAV = numpy.sum(abs(signalOrigin),axis=0)
        sumMAV = sumMAV.tolist()
        return sumMAV
    def sumWL(self,signalOrigin):
        sumWL = numpy.sum(abs(signalOrigin[0:299,:] - signalOrigin[1:,:])/299,axis=0)
        sumWL = sumWL.tolist()
        return sumWL
    def sumZC(self,signalOrigin):
        condition1 = (abs(signalOrigin[0:299,:]-signalOrigin[1:,:])>50)+0
        condition2 = (numpy.multiply(signalOrigin[0:299,:],signalOrigin[1:,:])<0)+0
        sumZC = numpy.sum(((condition1+condition2)>1)+0,axis=0)
        sumZC = sumZC.tolist()
        return sumZC
    def sumDASDV(self,signalOrigin):
        sumDASDV = numpy.sqrt(numpy.sum((signalOrigin[0:299,:] - signalOrigin[1:,:])**2/299,axis=0))
        sumDASDV = sumDASDV.tolist()
        return sumDASDV
    def sumLOG(self,signalOrigin):
        sumLOG = numpy.exp(numpy.sum(signalOrigin/300,axis=0))
        sumLOG = sumLOG.tolist()
        return sumLOG
    def sumSII(self,signalOrigin):
        sumSII = numpy.sum(signalOrigin**2,axis=0)
        sumSII = sumSII.tolist()
        return sumSII
    def sumTM3(self,signalOrigin):
        sumTM3 = abs(numpy.sum(signalOrigin**3/300,axis=0))
        sumTM3 = sumTM3.tolist()
        return sumTM3
    def sumTM4(self,signalOrigin):
        sumTM4 = abs(numpy.sum(signalOrigin**4/300,axis=0))
        sumTM4 = sumTM4.tolist()
        return sumTM4
    def sumTM5(self,signalOrigin):
        sumTM5 = abs(numpy.sum(signalOrigin**5/300,axis=0))
        sumTM5 = sumTM5.tolist()
        return sumTM5
    def frequencyRatio(self,signalOrigin):
        sign = abs(numpy.fft.fft(signalOrigin, 512,axis = 0) / 300)
        sumfrequencyRatio = numpy.min(sign, axis = 0)/numpy.max(sign, axis = 0)
        sumfrequencyRatio = sumfrequencyRatio.tolist()
        return sumfrequencyRatio
    def sumIEMG(self,signalOrigin):
        sumIEMG = numpy.sum(abs(signalOrigin)/300,axis=0)
        sumIEMG = sumIEMG.tolist()
        return sumIEMG
    def sumMFMN(self,signalOrigin):
        f = 1000*numpy.linspace(0,1,512)
        sign = abs(numpy.fft.fft(signalOrigin, 512,axis = 0) / 300)
        sumMFMN = numpy.sum(numpy.multiply(numpy.tile(f,(16,1)).T,sign),axis=0)/numpy.sum(sign,axis=0)
        sumMFMN = sumMFMN.tolist()
        return sumMFMN
    def sumMFMD(self,signalOrigin):
        sign = abs(numpy.fft.fft(signalOrigin, 512,axis = 0) / 300)
        sumMFMD = numpy.sum(sign, axis = 0)/2
        sumMFMD = sumMFMD.tolist()
        return sumMFMD
    def readFile(self):
        # print(time.ctime())
        trainOrigin = numpy.loadtxt('muscle1.txt')
        testOrigin = numpy.loadtxt('muscle2.txt')
        trainData = self.deal(trainOrigin)
        testData = self.deal(testOrigin)
        # print(trainData.shape)
        # print(testData.shape)
        # print(time.ctime())
        return trainData,testData
    def deal(self,trainOrigin):
        trainRow = trainOrigin.shape[0]
        trainData1= []
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
                trainData = []
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
                trainData+=RMS
                trainData+=MAV
                trainData+=WL
                trainData+=ZC
                trainData+=DASDV
                trainData+=LOG
                trainData+=SII
                trainData+=TM3
                trainData+=TM4
                trainData+=TM5
                trainData += frequencyRatio
                trainData += MFMN
                trainData += MFMD
                trainData += IEMG
                trainData1.append(trainData)
        trainData1 = numpy.array(trainData1)
        return trainData1
# A= dealxh()
# A.readFile()