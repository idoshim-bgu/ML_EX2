from softsvm import softsvm
from softsvmpoly import softsvmpoly
import numpy as np
import matplotlib.pyplot as plt

def _soft_svm_poly(params,trainX, trainY):
    return softsvmpoly(params[0], params[1], trainX, trainY)

def test_predictor(pred, testX, testY):
    count = 0
    for i in range(testX.shape[0]):
        if pred(np.sign(testX[i].dot(pred)) != testY[i]):
            count += 1
    return count / testY.shape[0]

def k_fold_validation(k, params,trainX, trainY,learning_algo):
    n_samples = trainX.shape[0]
    fold_size = n_samples // k
    param_error = []

    for param in params:
        errs = []
        for i in range(k):
            # Split the data into training and validation sets
            start = i * fold_size
            end = start + fold_size
            X_validation = trainX[start:end]
            y_validation = trainY[start:end]
            X_train = np.concatenate([trainX[:start], trainX[end:]])
            y_train = np.concatenate([trainY[:start], trainY[end:]])

            predictor = learning_algo(param,X_train,y_train)
            errs.append(test_predictor(predictor,X_validation,y_validation))
        param_error.append(np.mean(np.array(errs)))
    print(params[np.argmin(param_error)])
    return learning_algo(params[np.argmin(param_error)],trainX, trainY)

    
def get_graph_data():
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain']

    ret1 = []
    ret2 = []

    for i in range(len(trainX)):
        print(trainX[i])
        if trainy[i] == 1:
            ret1.append(trainX[i])
        else:
            ret2.append(trainX[i])
    
    return ret1, ret2

def main():
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain']
    params = np.array(np.meshgrid([1,10,100], [2,5,8])).T.reshape(-1, 2)
    k_fold_validation(5, params,trainX, trainy, _soft_svm_poly)
    

if __name__ == '__main__':
    main()