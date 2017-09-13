import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import math
import sys

def printHelp():
    print("do_ROC.py [file_locations] [max_number_clusters]")

if len(sys.argv) is not 3:
    printHelp()
    exit()

dirname = sys.argv[1]
num_clusters = sys.argv[2]

for i in range(2, int(num_clusters)+1):
    fprfile = dirname+"/{}_clusters_FPR.txt".format(i)
    tprfile = dirname+"/{}_clusters_TPR.txt".format(i)
    fpr_arr = np.genfromtxt(fprfile)
    tpr_arr = np.genfromtxt(tprfile)

    best_idx = 0
    old_dist = 2
    for j in range(1,len(fpr_arr)):
        dist = math.sqrt(fpr_arr[j] **2 + (1-tpr_arr[j]) **2)
        if(old_dist > dist):
            best_idx = j
            old_dist = dist
            
    print("Best index = {}, FPR = {}, TPR = {}, with {} clusters".format(best_idx, fpr_arr[best_idx], tpr_arr[best_idx], i))

    plotname = dirname + "/{}_cluster_ROC.png".format(i)
    plt.figure()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.plot(fpr_arr[:-1], tpr_arr[:-1], 'o', label='ROC at varied cutoffs', color='red')
    plt.plot(fpr_arr[best_idx], tpr_arr[best_idx], 's', label='Best cutoff', color='blue')
    plt.plot(fpr_arr, fpr_arr, label='totally random', ls=':', color='black')
    plt.legend(loc='best')
    plt.savefig(plotname)
