import numpy as np
from matplotlib import pyplot as plt
import sys

directory = sys.argv[1]
min_num_hills = int(sys.argv[2])
max_num_hills = int(sys.argv[3])

fpr_arr, tpr_arr, accuracy_arr = [], [], []
x_axis = range(min_num_hills, max_num_hills+1)

for i in range(min_num_hills, max_num_hills+1):
    with open('{}/{}_clusters_ROC_log.txt'.format(directory, i)) as f:
        lines = f.readlines()
        fpr_arr.append(lines[1].split()[2])
        tpr_arr.append(lines[2].split()[2])
        accuracy_arr.append(lines[3].split()[1])

plt.figure()
plt.title('Statistics for Various Kernel Numbers')
plt.xlabel('Number of Gaussian Kernels')
plt.ylabel('Statistic Value')
plt.grid(color='grey', linestyle='--')
plt.plot(x_axis, fpr_arr, 'o', color = 'red', ls='--', label='FPR')
plt.plot(x_axis, tpr_arr, 'o', color = 'green', ls='--',label='TPR')
plt.plot(x_axis, accuracy_arr, 's', color='blue', ls='--',label='Accuracy')
plt.legend(loc='best')
plt.savefig('{}/statistics.svg'.format(directory))
