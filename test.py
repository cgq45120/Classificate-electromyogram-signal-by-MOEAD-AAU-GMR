import numpy as np
import math
import random
if __name__ == '__main__':
    # a = np.array([[2,3,4],[5,6,7]])
    b = np.array([1,2,3,2,1,3])
    print(b.shape)
    c = np.argsort(b)
    d = [a[3]for i in c]
    # for i in c:
    #     print(b[i])
    # a = np.arange(1,21,1).reshape((4,5))
    # print(a)
    # print(a[:,1]*a[:,2])
    # print(a[:,-4:].min(0))
    # a = np.append(a,b,axis =0)
    # print(a)
    # print(a)
    # a.extend(b)
    # print(a)
