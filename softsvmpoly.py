import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt

def predict_with_alpha(alpha, trainX, k):
    def ret_func(x):
        ret = 0
        for i in range(trainX.shape[0]):
            ret += alpha[i] * poly_kernel(trainX[i], x, k)
        return np.sign(ret)
    return ret_func
    

def poly_kernel(x1,x2, k):
    return (x1.dot(x2) + 1)**k

def create_gramm(trainX, k):
    return np.array([[poly_kernel(x1,x2,k) for x1 in trainX] for x2 in trainX])

# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvmpoly(l: float, k: int, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    G = create_gramm(trainX,k)
    d = len(trainX[0])
    m = len(trainX)
    u = np.append(np.zeros(m), np.full(m, 1/m))
    v = np.append(np.zeros(m), np.ones(m))
    H = np.block([
                [G * 2*l, np.zeros((m,m))],
                [ np.zeros((m,m)),  np.zeros((m,m))]
                ])
    H = H + np.identity(H.shape[0]) * np.exp(-10)
    yGi = np.array([trainy[i] * G[i] for i in range(m)])
    A = np.block([
        [np.zeros((m,m)), np.identity(m)],
        [yGi, np.identity(m)]
    ])
    solvers.options['show_progress'] = False
    sol = solvers.qp(matrix(H), matrix(u), -matrix(A), -matrix(v))
    return np.array(sol["x"][:m])

    """
    for quadratic problem:
        1/2 z^T*H*z + <u,z>
        Az>=v

        z = {a1,...,am,xi1,...,xim}
        H = block matrix 2mX2m of
            {
                2lG   0
                0   0
            }
        u = {0,...m...,0,1/m,...m..., 1/m }
        v = {0,...m...,0,1,...m..., 1}
        A = block matrix 2mX2m of
        {
            0          Im
            {yi*G[i]}    Im
        }
    """


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvmpoly algorithm
    w = softsvmpoly(10, 5, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 4
