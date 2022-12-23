import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def softsvm(l, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    d = len(trainX[0])
    m = len(trainX)
    u = np.append(np.zeros(d), np.full(m, 1/m))
    v = np.append(np.zeros(m), np.ones(m))
    H = np.block([
                [np.identity(d) * 2*l, np.zeros((d,m))],
                [ np.zeros((m,d)),  np.zeros((m,m))]
                ])
    H = H + np.identity(H.shape[0]) * np.exp(-10)
    yXi = np.array([trainy[i] * trainX[i] for i in range(m)])
    A = np.block([
        [np.zeros((m,d)), np.identity(m)],
        [yXi, np.identity(m)]
    ])
    solvers.options['show_progress'] = False
    sol = solvers.qp(matrix(H), matrix(u), -matrix(A), -matrix(v))


    """
    for quadratic problem:
        1/2 z^T*H*z + <u,z>
        Az>=v

        z = {w1,...,wd,xi1,...,xim}
        H = block matrix d+mXd+m of
            {
                2l*Id   0
                0       0
            }
        u = {0,...d...,0,1/m,...m..., 1/m }
        v = {0,...m...,0,1,...m..., 1}
        A = block matrix m+dX2m of
        {
            0          Im
            {yi*xi}    Im
        }
    """

    return np.array(sol["x"][:d])

def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # count = 0
    # for i in range(len(testX)):
    #     if (np.sign(testX[i] @ w)[0] == np.sign(testy[i])):
    #         count +=1
    # print(count/len(testy))

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")




if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
