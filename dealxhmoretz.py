import numpy
import math
import time
class dealxh(object):
    def sumRMS(self,signalOrigin,col):
        sumRms = []
        for i in range(col):
            sumfinal = 0
            for j in range(300):
                sumfinal += signalOrigin[j,i]**2
            sumfinal = math.sqrt(sumfinal/300)
            sumRms.append(sumfinal)
        return sumRms
    def sumMAV(self,signalOrigin,col):
        sumMAV = []
        for i in range(col):
            sumfinal = 0
            for j in range(300):
                sumfinal += abs(signalOrigin[j,i])
            sumMAV.append(sumfinal)
        return sumMAV
    def sumWL(self,signalOrigin,col):
        sumWL = []
        for i in range(col):
            sumfinal = 0
            for j in range(299):
                sumfinal +=abs(signalOrigin[j,i]-signalOrigin[j+1,i])
            sumfinal = sumfinal/299
            sumWL.append(sumfinal)
        return sumWL
    def sumZC(self,signalOrigin,col):
        sumZC = []
        for i in range(col):
            sumfinal = 0
            for j in range(299):
                if abs(signalOrigin[j,i]-signalOrigin[j+1,i])>50 and signalOrigin[j,i]*signalOrigin[j+1,i]<0:
                    sumfinal = sumfinal+1
            sumZC.append(sumfinal)
        return sumZC
    def sumDASDV(self,signalOrigin,col):
        sumDASDV = []
        for i in range(col):
            sumfinal = 0
            for j in range(299):
                sumfinal +=(signalOrigin[j,i]-signalOrigin[j+1,i])**2
            sumfinal = sumfinal/299
            sumfinal = math.sqrt(sumfinal)
            sumDASDV.append(sumfinal)
        return sumDASDV
    def sumLOG(self,signalOrigin,col):
        sumLOG = []
        for i in range(col):
            sumfinal = 0
            for j in range(300):
                sumfinal += signalOrigin[j, i]
            sumfinal = math.exp(sumfinal/300)
            sumLOG.append(sumfinal)
        return sumLOG
    def sumSII(self,signalOrigin,col):
        sumSII = []
        for i in range(col):
            sumfinal = 0
            for j in range(300):
                sumfinal += signalOrigin[j,i]**2
            sumSII.append(sumfinal)
        return sumSII
    def sumTM3(self,signalOrigin,col):
        sumTM3 = []
        for i in range(col):
            sumfinal = 0
            for j in range(300):
                sumfinal += signalOrigin[j, i] ** 3
            sumfinal = abs(sumfinal/300)
            sumTM3.append(sumfinal)
        return sumTM3
    def sumTM4(self,signalOrigin,col):
        sumTM4 = []
        for i in range(col):
            sumfinal = 0
            for j in range(300):
                sumfinal += signalOrigin[j, i] ** 4
            sumfinal = abs(sumfinal/300)
            sumTM4.append(sumfinal)
        return sumTM4
    def sumTM5(self,signalOrigin,col):
        sumTM5 = []
        for i in range(col):
            sumfinal = 0
            for j in range(300):
                sumfinal += signalOrigin[j, i] ** 5
            sumfinal = abs(sumfinal/300)
            sumTM5.append(sumfinal)
        return sumTM5
    def sumnewRMS(self,signalOrigin,col):
        sumnewRms = []
        for i in range(col):
            sumfinal = 0
            for j in range(300):
                sumfinal += signalOrigin[j, i]
            sumfinal = math.sqrt(sumfinal / 300)
            sumnewRms.append(sumfinal)
        return sumnewRms
    def sumSampEn(self,signalOrigin,col):
        sumSampEn = []
        r = []
        for i in range(col):
            k = numpy.var(signalOrigin[:,i])
            k = 0.2*math.sqrt(k)
            r.append(k)
        for i in range(col):
            out = self.SampEn(2,r[i],signalOrigin[:,i],1)
            sumSampEn.append(out)
        return sumSampEn
    def SampEn(self,dim, r, data, tau):
        N = len(data)
        correl = numpy.zeros((1,2))
        dataMat = numpy.zeros((3,298))
        for i in range(dim+1):
            dataMat[i,:] = data[i:N+i-dim]
        for i in range(dim,dim+2):
            count = numpy.zeros((1,N-dim))
            tempMat = dataMat[0:i,:]
            for j in range(N-i-1):
                dist1 = numpy.array(numpy.tile((tempMat[:,j]),(1,N-dim-j-1))).reshape(N-dim-j-1,i).T
                dist = abs(tempMat[:,j+1:N-dim]-dist1).max(0)
                sum1=0
                for k in range(N-dim-j-1):
                    if dist[k]<r:
                        sum1+=1
                count[0,j] = sum1/(N-dim)
            correl[0,i-dim] = sum(sum(count))/(N-dim)
        saen = math.log(correl[0,0]/correl[0,1])
        return saen


    def frequencyRatio(self,signalOrigin,col):
        sumfrequencyRatio = []
        for i in range(col):
            sign = abs(numpy.fft.fft(signalOrigin[:,i],512)/300)
            sumfrequencyRatio.append(min(sign)/max(sign))
        return sumfrequencyRatio
    def sumIEMG(self,signalOrigin,col):
        sumIEMG = []
        for i in range(col):
            sumfinal = 0
            for j in range(300):
                sumfinal += abs(signalOrigin[j, i])
            sumIEMG.append(sumfinal/300)
        return sumIEMG
    def sumMFMN(self,signalOrigin,col):
        sumMFMN = []
        f = 1000*numpy.linspace(0,1,512)
        for i in range(col):
            sign = abs(numpy.fft.fft(signalOrigin[:, i], 512) / 300)
            sumfinal = sum(numpy.multiply(sign,f.T))/sum(sign)
            sumMFMN.append(sumfinal)
        return sumMFMN
    def sumMFMD(self,signalOrigin,col):
        sumMFMD = []
        for i in range(col):
            sign = abs(numpy.fft.fft(signalOrigin[:,i],512)/300)
            sumMFMD.append(sum(sign)/2)
        return sumMFMD
    def readFile(self):
        # print(time.ctime())
        trainOrigin = numpy.loadtxt('F:\\muscle1.txt')
        testOrigin = numpy.loadtxt('F:\\muscle2.txt')
        trainData = self.deal(trainOrigin)
        testData = self.deal(testOrigin)
        # print(trainData.shape)
        # print(testData.shape)
        # print(time.ctime())
        return trainData,testData
    def deal(self,trainOrigin):
        trainRow = trainOrigin.shape[0]
        col = trainOrigin.shape[1]
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
                RMS = self.sumRMS(trainOrigin[i*50:j,:],col)
                MAV = self.sumMAV(trainOrigin[i*50:j,:],col)
                WL = self.sumWL(trainOrigin[i*50:j,:],col)
                ZC = self.sumZC(trainOrigin[i*50:j,:],col)
                DASDV = self.sumDASDV(trainOrigin[i*50:j,:],col)
                LOG = self.sumLOG(trainOrigin[i*50:j,:],col)
                SII = self.sumSII(trainOrigin[i*50:j,:],col)
                TM3 = self.sumTM3(trainOrigin[i*50:j,:],col)
                TM4 = self.sumTM4(trainOrigin[i*50:j,:],col)
                TM5 = self.sumTM5(trainOrigin[i*50:j,:],col)
                frequencyRatio = self.frequencyRatio(trainOrigin[i*50:j,:],col)
                MFMN = self.sumMFMN(trainOrigin[i*50:j,:],col)
                MFMD = self.sumMFMD(trainOrigin[i*50:j,:],col)
                IEMG = self.sumIEMG(trainOrigin[i*50:j,:],col)
                SampEn = self.sumSampEn(trainOrigin[i*50:j,:],col)
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
                trainData += SampEn
                trainData1.append(trainData)
        trainData1 = numpy.array(trainData1)
        return trainData1
A= dealxh()
A.readFile()