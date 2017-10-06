import numpy as np
from matplotlib import pyplot as plt
import sys

directory = sys.argv[1]
num_classes = int(sys.argv[2])
min_length = int(sys.argv[3])
max_length = int(sys.argv[4])

fpr_arr, tpr_arr, accuracy_arr = [], [], []
x_axis = range(min_length, max_length+1)

for j in range(min_length, max_length+1):
    with open('{}/gpos_{}_classes_length_{}/{}_classes_length_{}_ROC_log.txt'.format(directory, num_classes, j, num_classes, j)) as f:
        lines = f.readlines()
        fpr_arr.append(lines[1].split()[2])
        tpr_arr.append(lines[2].split()[2])
        accuracy_arr.append(lines[3].split()[1])

plt.figure()
plt.title('Statistics for Various Motif Lengths, {} Classes'.format(num_classes))
plt.xlabel('Motif Length')
plt.ylabel('Fraction')
plt.grid(color='grey', linestyle='--')
plt.plot(x_axis, fpr_arr, 'o', color = 'red', ls='--', label='FPR')
plt.plot(x_axis, tpr_arr, 'o', color = 'green', ls='--',label='TPR')
plt.plot(x_axis, accuracy_arr, 's', color='blue', ls='--',label='Accuracy')
plt.legend(loc='best')
plt.savefig('{}/motif_length_{}_statistics.svg'.format(directory, num_classes))
plt.savefig('{}/motif_length_{}_statistics.pdf'.format(directory, num_classes))
plt.savefig('{}/motif_length_{}_statistics.png'.format(directory, num_classes))
