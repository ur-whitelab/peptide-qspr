import numpy as np
from matplotlib import pyplot as plt
import sys
import math

#this script is just for making the small plots for the paper

def printhelp():
    print("Usage: do_gibbs_ROC.py [root_directory] [num_classes] [motif_length]")
    exit(1)

if len(sys.argv) != 4:
    printhelp()

DIRNAME = sys.argv[1]
NUM_MOTIF_CLASSES = int(sys.argv[2])
MOTIF_LENGTH = int(sys.argv[3])

ALPHABET = ['A','R','N','D','C','Q','E','G','H','I',
            'L','K','M','F','P','S','T','W','Y','V']


DATA_DIR = '/home/rainier/pymc3_qspr/data/'


for i in range(NUM_MOTIF_CLASSES):
    for j in range(MOTIF_LENGTH):
        histo = np.genfromtxt('{}/class_{}_of_{}_position_{}_motif_dist.txt'.format(DIRNAME, i, NUM_MOTIF_CLASSES, j))
        Alist = []
        lilhist = []
        for k in range(len(ALPHABET)):
            if(histo[k] > 0.05):#cutoff at 5% prob
                Alist.append(ALPHABET[k])
                lilhist.append(histo[k])
        x_axis = np.linspace(0, len(Alist)*0.05, len(Alist))
        plt.figure()
        plt.ylim(0,1.0)
        plt.xlim(-0.05, len(Alist)*0.1)
        plt.bar(x_axis, lilhist, width=0.05)
        plt.xticks(x_axis, ALPHABET)
        plt.savefig('{}/c{}of{}p{}_motif_dist.png'.format(DIRNAME, i, NUM_MOTIF_CLASSES, j))
        plt.savefig('{}/c{}of{}p{}_motif_dist.pdf'.format(DIRNAME, i, NUM_MOTIF_CLASSES, j))
        plt.savefig('{}/c{}of{}p{}_motif_dist.svg'.format(DIRNAME, i, NUM_MOTIF_CLASSES, j))
        plt.close()
