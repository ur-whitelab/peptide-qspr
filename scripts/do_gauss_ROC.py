import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import math
import sys

'''This file iterates over a specified lower and upper bound of Gaussian mixture cluster-numbers, 
and calculates the ROC curve for each one, saving the resulting plots. It assumes the files
 for each kernel number exist and are in the same directory!'''

def printHelp():
    print("Usage: do_gauss_ROC.py [file_locations] [min_number_clusters] [max_number_clusters] [weight_for_FPR_dist]")

if len(sys.argv) is not 5:
    printHelp()
    exit()

dirname = sys.argv[1]
min_num_clusters = sys.argv[2]
max_num_clusters = sys.argv[3]
prefactor = float(sys.argv[4])


for i in range(int(min_num_clusters), int(max_num_clusters)+1):
    fprfile = dirname+"/{}_clusters_FPR.txt".format(i)
    tprfile = dirname+"/{}_clusters_TPR.txt".format(i)
    fpr_arr = np.genfromtxt(fprfile)
    tpr_arr = np.genfromtxt(tprfile)

    best_idx = 0
    old_dist = 2
    for j in range(1,len(fpr_arr)):
        dist = math.sqrt(prefactor*fpr_arr[j] **2 + (1-tpr_arr[j]) **2)
        if(old_dist > dist):
            best_idx = j
            old_dist = dist
            
    print("Best index = {}, FPR = {}, TPR = {}, with {} clusters".format(best_idx, fpr_arr[best_idx], tpr_arr[best_idx], i))

    plotname = dirname + "/{}_cluster_NEW_ROC.".format(i)
    plt.rcParams.update({'font.size': 7})
    plt.figure(figsize = (2.5, 2.0), dpi=800)
    plt.xlabel('FPR')#, labelpad=-12)
    plt.ylabel('TPR')#, labelpad=-18)
    #plt.title('{} Clusters ROC Curve'.format(i))
    plt.plot(fpr_arr[:-1], tpr_arr[:-1], 'o', label='ROC at varied cutoffs', color='red', lw=2.0, ms=2.0)
    plt.plot(fpr_arr[best_idx], tpr_arr[best_idx], 's', label='Best cutoff', color='blue', lw=2.0, ms=2.0)
    plt.plot(fpr_arr, fpr_arr, label='totally random', ls=':', color='black', lw=2.0, ms=2.0)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig(plotname+'png')
    plt.savefig(plotname+'pdf')
    plt.savefig(plotname+'svg')

    best_tpr = tpr_arr[best_idx]
    best_fpr = fpr_arr[best_idx]
    accuracy = (best_tpr + (1.0-best_fpr))/2.0
    with open('{}/{}_clusters_ROC_log.txt'.format(dirname, i), 'w+') as f:
        f.write('best idx {}\n'.format(best_idx))
        f.write('best FPR {}\n'.format(fpr_arr[best_idx]))
        f.write('best TPR {}\n'.format(tpr_arr[best_idx]))
        f.write('accuracy {}\n'.format(accuracy))
