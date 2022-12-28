from softsvmpoly import softsvmpoly
import numpy as np
import matplotlib.pyplot as plt

def to_poly_feature_space(x,k):
    degrees = sequences(x.shape[0],k)
    ret =[]
    for deg in degrees:
        a = 1
        for i in range(x.shape[0]):
            a *= x[i]**deg[i]
        deg.append(k - sum(deg))
        a *= np.sqrt(multinomial(deg))
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

def multinomial(lst):
    res, i = 1, sum(lst)
    i0 = lst.index(max(lst))
    for a in lst[:i0] + lst[i0+1:]:
        for j in range(1,a+1):
            res *= i
            res //= j
            i -= 1
    return res

def w_from_alpha(alpha, trainX, k):
    w = np.array([])
    poly_trainX = np.apply_along_axis(to_poly_feature_space,1,trainX,k)
    return  np.matmul(alpha.T, poly_trainX)

def get_graph_data(testX,w,k, testY):
    ret1 = []
    ret2 = []

    count = 0
    for i in range(testX.shape[0]):
        if np.sign(np.dot(w,to_poly_feature_space(testX[i],k))) > 0:
            ret1.append(testX[i])
        else:
            ret2.append(testX[i])
        
        if np.sign(np.dot(w,to_poly_feature_space(testX[i],k))) != np.sign(testY[i]):
            count += 1
    print(count/testY.shape[0])
    
    return np.array(ret1), np.array(ret2)

def main():
    k, l = 5, 1
    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain']
    testX = np.concatenate((data['Xtest'],trainX))
    testY = np.append(data['Ytest'],trainy)
    alpha = softsvmpoly(l, k,trainX,trainy)
    w = w_from_alpha(alpha,trainX,k)
    label1, label2 = get_graph_data(testX, w, k,testY)

    plt.style.use('seaborn-whitegrid')
    plt.plot(label1[:, 0],label1[:, 1] ,"o", color="blue")
    plt.plot(label2[:, 0],label2[:, 1] ,"o", color="red")

    plt.show()
    

if __name__ == '__main__':
    main()