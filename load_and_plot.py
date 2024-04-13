import numpy as np
import matplotlib.pyplot as plt
import shutil
import os

env = "LunarLander"
#percent = "0-1"

for p in [1.0, 0.1, 0.5, 0.0]:
    percent = str(p).replace(".", "-")

    true_path = f"random_comp_plots/true_y_{env}_{percent}.npy"
    if not os.path.exists(true_path):
        shutil.copyfile(f'algorithms/finetune/true_y_{env}_{percent}.npy', true_path)
    true_y = np.load(true_path)

    pred_path = f"random_comp_plots/pred_y_{env}_{percent}.npy"
    if not os.path.exists(pred_path):
        shutil.copyfile(f'algorithms/finetune/pred_y_{env}_{percent}.npy', f"random_comp_plots/pred_y_{env}_{percent}.npy")
    pred_y = np.load(pred_path)

    print("Num Samples: ", len(true_y))
    lines = {'linestyle': 'None'}
    plt.rc('lines', **lines)

    print(f"Mean State Value: {np.round(np.mean(pred_y[:,1]),2)} +\- {np.round(np.mean(pred_y[:,2]),2)}")

    x=0
    max_steps_per_plot = min(len(pred_y)+1, 100)
    while x < len(pred_y):
        for i in range(x, x+max_steps_per_plot):
            if i == len(pred_y):
                break
            plt.ylim(-125, 175)
            if i == x:
                plt.errorbar(pred_y[i,0].astype('int'), pred_y[i,1], pred_y[i,2], color="blue", marker='.', label="Prediction", fmt='')
                plt.scatter(true_y[i,0].astype('int'), true_y[i,1], facecolor="None", edgecolor="red", label="Sample")
            else:
                plt.errorbar(pred_y[i,0].astype('int'), pred_y[i,1], pred_y[i,2], color="blue", marker='.', fmt='')
                plt.scatter(true_y[i,0].astype('int'), true_y[i,1], facecolor="None", edgecolor="red")
            
        plt.xlabel("State Label")
        plt.ylabel("State Value")
        plt.legend()
        plt.savefig(f"random_comp_plots/state_pred_results_{i}_{percent}.png")
        #plt.show()
        plt.close()
        x+=max_steps_per_plot
