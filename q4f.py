from softsvmpoly import softsvmpoly, predict_with_alpha
from softsvm import softsvm
import numpy as np
import matplotlib.pyplot as plt

def to_poly_feature_space(x,k):
    degrees = sequences(x.shape[0],k)
    ret =[]
    for deg in degrees:
        a = 1
        for i in range(x.shape[0]):
            a *= x[i]**deg[i]
        ret.append(a)
    return np.array(ret)

def sequences(d, k):
    if d == 0:
        return [[]]
    else:
        result = []
        for i in range(k + 1):
            for seq in sequences(d - 1, k - i):
                result.append([i] + seq)
        return result


def main():
    k, l = 5, 1
    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain']
    alpha = softsvmpoly(l, k,trainX,trainy)
    

    

if __name__ == '__main__':
    main()