import numpy as np
import matplotlib.pyplot as plt

container = "982dcd8a9455"

true_y = np.load(f'plots/{container}_true_y.npy')
pred_y = np.load(f'plots/{container}_pred_y.npy')

lines = {'linestyle': 'None'}
plt.rc('lines', **lines)

x=0
while x < len(pred_y)-100:
    for i in range(x, x+100):
        if i == x:
            plt.errorbar(pred_y[i,0].astype('int'), pred_y[i,1], pred_y[i,2], color="blue", marker='.', label="Prediction", fmt='')
            plt.scatter(true_y[i,0].astype('int'), true_y[i,1], facecolor="None", edgecolor="red", label="Sample")
        else:
            plt.errorbar(pred_y[i,0].astype('int'), pred_y[i,1], pred_y[i,2], color="blue", marker='.', fmt='')
            plt.scatter(true_y[i,0].astype('int'), true_y[i,1], facecolor="None", edgecolor="red")
    plt.xlabel("State Label")
    plt.ylabel("State Value")
    plt.legend()
    plt.savefig(f"state_pred_results_{i}.png")
    plt.show()
    x+=100
