from softsvmpoly import softsvmpoly,predict_with_alpha
import numpy as np
import matplotlib.pyplot as plt


def get_graph_data(ks):
    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain']
    all_grids = []

    for k in ks:
        alphas = softsvmpoly(100,k, trainX, trainy)
        predictor = predict_with_alpha(alphas, trainX, k)
        grid = []

        for j in np.arange(-1,1,0.005):
            temp = []
            for i in np.arange(-1,1,0.005):
                if predictor(np.array([i,j])) > 0:
                    temp.append([255,0,0])
                else:
                    temp.append([0,0,255])
            grid.append(temp)
        all_grids.append(np.array(grid))


    return all_grids

def main():
    ks = [3,5,8]
    grids = get_graph_data(ks)

    figure, axis = plt.subplots(1,3)
    for i in range(len(grids)):
        axis[i].imshow(grids[i], origin='lower', extent=[-1, 1, -1, 1],cmap="rainbow" )
        axis[i].set_title(f"k = {ks[i]}")

    plt.show()
    

if __name__ == '__main__':
    main()