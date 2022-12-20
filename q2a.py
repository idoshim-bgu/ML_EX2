from softsvm import softsvm
import numpy as np
import matplotlib.pyplot as plt


def get_graph_data(train_sizes, lambdas):
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testY = data['Ytest']

    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:train_sizes]]
    _trainy = trainy[indices[:train_sizes]]

    error_data = []
    err_mins =[]
    err_maxs = []

    for l in lambdas:
        errs = []
        for i in range(10):
            w = softsvm(l, _trainX, _trainy)

            indices = np.random.permutation(testX.shape[0])
            _testX = testX[indices[:train_sizes]]
            _testY = testY[indices[:train_sizes]]

            count = 0
            for i in range(_testY.shape[0]):
                if np.sign(np.dot(_testX[i] , w)) != np.sign(_testY[i]):
                    count += 1
            errs.append(count/_testY.shape[0])

        error_data.append(np.mean(np.array(errs)))
        err_maxs.append(max(errs) - error_data[-1])
        err_mins.append(error_data[-1] - min(errs))

    
    return error_data, err_mins, err_maxs

def main():
    sample_size = 100
    lambdas = [10**i for i in range(1,10)]
    errors, err_mins, err_maxs = get_graph_data(sample_size, lambdas)

    plt.style.use('seaborn-whitegrid')
    plt.xlabel("lambda")
    plt.ylabel("mean error")
    plt.errorbar([f"10^{i}" for i in range(1,10)], errors, yerr=[err_mins,err_maxs], fmt='o')

    plt.show()
    

if __name__ == '__main__':
    main()