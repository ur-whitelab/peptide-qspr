import numpy as np
from matplotlib import pyplot as plt
import sys
import math

def printhelp():
    print("Usage: plot_motif_dists.py [root_directory] [num_classes] [motif_length]")
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
        plt.rcParams.update({'font.size': 7})
        plt.figure(figsize = (2.5, 2.0), dpi=800)
        plt.title('Class {} of {} Position {} Motif Distribution'.format(i+1, NUM_MOTIF_CLASSES, j+1))
        plt.xlabel('Amino Acid')
        plt.ylabel('Probability')
        plt.ylim(0, 1.0)
        x_axis = list(range(len(ALPHABET)))
        histo = np.genfromtxt('{}/class_{}_of_{}_position_{}_motif_dist.txt'.format(DIRNAME, i, NUM_MOTIF_CLASSES, j))
        plt.bar(x_axis, histo)
        plt.xticks(x_axis, ALPHABET)
        plt.tight_layout()
        plt.savefig('{}/class_{}_of_{}_position_{}_motif_dist.png'.format(DIRNAME, i, NUM_MOTIF_CLASSES, j))
        plt.savefig('{}/class_{}_of_{}_position_{}_motif_dist.pdf'.format(DIRNAME, i, NUM_MOTIF_CLASSES, j))
        plt.savefig('{}/class_{}_of_{}_position_{}_motif_dist.svg'.format(DIRNAME, i, NUM_MOTIF_CLASSES, j))
