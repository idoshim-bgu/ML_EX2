from softsvm import softsvm
import numpy as np
import matplotlib.pyplot as plt


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
    label1, label2 = get_graph_data()
    label1 = np.array(label1)
    label2 = np.array(label2)
    plt.style.use('seaborn-whitegrid')
    plt.plot(label1[:, 0],label1[:, 1] ,"o" , color="blue")
    plt.plot(label2[:, 0],label2[:, 1] ,"o", color="red")
    plt.legend(["label 1", "label -1"])

    plt.show()
    

if __name__ == '__main__':
    main()