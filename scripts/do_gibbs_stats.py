import numpy as np
from matplotlib import pyplot as plt
import sys
import os

'''This file iterates over a specified lower and upper bound of Gaussian mixture cluster-numbers, 
and plots FPR, TPR, and accuracy for each one. It assumes the files for each kernel number
 exist and are in the same directory!'''

number of motif classes. Assumes all that class's data subdirs are in the same directory given.'''

def printHelp():
    print("Usage: do_gibbs_stats.py [directory] [num_motifs] [min_motif_length] [max_motif_length] [HUMAN? (default: False)]\nInput 'HUMAN' as True if we're doing human datasets, otherwise defaults to doing gram+.")
    exit(1)

if( len(sys.argv) != 5 and len(sys.argv) != 6):
    printHelp()

HUMAN = False
    
directory = sys.argv[1]
num_classes = int(sys.argv[2])
min_length = int(sys.argv[3])
max_length = int(sys.argv[4])
if(len(sys.argv) == 6):
    HUMAN = bool(sys.argv[5])
    
    

fpr_arr, tpr_arr, accuracy_arr = [], [], []
x_axis = range(min_length, max_length+1)

if(HUMAN):
    prefactor = 'human'
else:
    prefactor = 'gpos'

for j in range(min_length, max_length+1):
    fname = '{}/{}_{}_classes_length_{}/{}_classes_length_{}_ROC_log.txt'.format(directory, prefactor, num_classes, j, num_classes, j)
    if(os.path.isfile(fname)):
        with open(fname) as f:
            lines = f.readlines()
            fpr_arr.append(lines[1].split()[2])
            tpr_arr.append(lines[2].split()[2])
            accuracy_arr.append(lines[3].split()[1])
    else:
        fpr_arr.append(None)
        tpr_arr.append(None)
        accuracy_arr.append(None)
fpr_arr = np.array(fpr_arr).astype(np.double)
tpr_arr = np.array(tpr_arr).astype(np.double)
accuracy_arr = np.array(accuracy_arr).astype(np.double)

fprmask = np.isfinite(fpr_arr)
tprmask = np.isfinite(tpr_arr)
accmask = np.isfinite(accuracy_arr)

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
