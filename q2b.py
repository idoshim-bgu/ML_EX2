from softsvm import softsvm
import numpy as np
import matplotlib.pyplot as plt


def get_graph_data(train_sizes, lambdas, n):
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testY = data['Ytest']

    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:train_sizes]]
    _trainY = trainy[indices[:train_sizes]]

    error_data = []
    err_mins =[]
    err_maxs = []
    train_errs = []

    for l in lambdas:
        test_errs = []
        w = softsvm(l, _trainX, _trainY)
        count_train = 0

        for j in range(n):
            indices = np.random.permutation(testX.shape[0])
            _testX = testX[indices[:1000]]
            _testY = testY[indices[:1000]]

            count_test = 0
            for i in range(_trainY.shape[0]):
                if np.sign(np.dot(_testX[i] , w)) != np.sign(_testY[i]):
                    count_test += 1
                if j == 0:
                    if np.sign(np.dot(_trainX[i] , w)) != np.sign(_trainY[i]):
                        count_train += 1
            test_errs.append(count_test/_trainY.shape[0])
            if j ==0:
                train_errs.append(count_train/_trainY.shape[0])
        

        error_data.append(np.mean(np.array(test_errs)))
        err_maxs.append(max(test_errs) - error_data[-1])
        err_mins.append(error_data[-1] - min(test_errs))

    
    return error_data, err_mins, err_maxs, train_errs

def main():
    lambdas = [10**i for i in range(1,11)]
    test_errors, err_mins, err_maxs, train_error = get_graph_data(100, lambdas, 10)

    plt.style.use('seaborn-whitegrid')
    plt.xlabel("lambda")
    plt.ylabel("mean error")
    plt.title("mean error versus lambda")
    plt.errorbar([f"10^{i}" for i in range(1,11)], test_errors, yerr=[err_mins,err_maxs], label="test sample 100", color="tab:blue")
    plt.plot( train_error, label="train sample 100", color="tab:green")
    
    
    lambdas = [10**i for i in [1,3,5,8]]
    test_errors, err_mins, err_maxs, train_error = get_graph_data(1000, lambdas, 1)
    plt.plot([f"10^{i}" for i in [1,3,5,8]], train_error, "o", label="train sample 1000", color="tab:olive")
    plt.plot([f"10^{i}" for i in [1,3,5,8]], test_errors, "o", label="test sample 1000", color="tab:cyan")

    #get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    #specify order of items in legend
    order = [3,0,2,1]

    #add legend to plot
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="upper left")

    plt.show()
    

if __name__ == '__main__':
    main()