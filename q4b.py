from softsvmpoly import softsvmpoly, predict_with_alpha
from softsvm import softsvm
import numpy as np
import matplotlib.pyplot as plt

def _soft_svm_poly(params, trainX, trainY):
    alphas = softsvmpoly(params[0], params[1], trainX, trainY)
    return predict_with_alpha(alphas,trainX,params[1])

def test_softsvmpoly(pred, testX, testY):
    count = 0
    for i in range(testX.shape[0]):
        if pred(testX[i]) != np.sign(testY[i]):
            count += 1
    return count / testY.shape[0]

def test_softsvm(pred, testX, testY):
    count = 0
    for i in range(testX.shape[0]):
        if np.sign(np.dot(testX[i], pred)) != np.sign(testY[i]):
            count += 1
    return count / testY.shape[0]

def k_fold_validation(k, params,trainX, trainY,learning_algo, test_pred):
    """
    :param k: number of folds.
    :param params: a list of different parameter sets to test.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :param learning_algo: a learning algorithm function that takes (param, trainX, trainY) and returns a predictor.
    :param test_pred: a function to test the returned predictor, takes (pred, testX, testY) and returns error value.
    :return: best predictor found with selected parameters.
    """
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
            errs.append(test_pred(predictor,X_validation,y_validation))
        param_error.append(np.mean(np.array(errs)))
        print(f"The average validation error for {param} is {param_error[-1]}")
    print(f"The selected pair is {params[np.argmin(param_error)]}")
    return learning_algo(params[np.argmin(param_error)],trainX, trainY)

def main():
    k = 5
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain']
    testY = data['Ytest']
    testX = data['Xtest']
    print("results for k fold cross validation on soft SVM with poly kernel:")
    params = np.array(np.meshgrid([1,10,100], [2,5,8])).T.reshape(-1, 2)
    predictor = k_fold_validation(k, params,trainX, trainy, _soft_svm_poly, test_softsvmpoly)
    print(f"test error for selected parameters: {test_softsvmpoly(predictor,testX,testY)}")

    print("results for k fold cross validation on soft SVM:")
    params = np.array([1,10,100])
    predictor = k_fold_validation(k, params,trainX, trainy, softsvm, test_softsvm)
    print(f"test error for selected parameters: {test_softsvm(predictor,testX,testY)}")

if __name__ == '__main__':
    main()