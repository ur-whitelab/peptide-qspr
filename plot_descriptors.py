import sys
import numpy as np
from matplotlib import pyplot as plt

def printHelp():
    print("Usage: plot_descriptors.py [directory] [min_num_clusters] [max_num_clusters] [output_dir]")
    exit(1)

if len(sys.argv) != 5:
    printHelp()

DIRNAME = sys.argv[1]
min_num_clusters = int(sys.argv[2])
max_num_clusters = int(sys.argv[3])
OUTDIR = sys.argv[4]


keys = ['netCharge', 'nChargedGroups', 'nNonPolarGroups']




for i in range(min_num_clusters, max_num_clusters+1):
    for key in keys:
            bins = np.genfromtxt('{}/{}_clusters_{}_observed.bins'.format(DIRNAME, i, key))
            counts = np.genfromtxt('{}/{}_clusters_{}_observed.counts'.format(DIRNAME, i, key))
            plt.figure()
            plt.title('{} Clusters {} Histogram'.format(i, key))
            plt.bar(bins[1:], height=counts/np.sum(counts))
            plt.xlabel('Value')
            plt.ylabel('Fraction')
            plt.savefig('{}/{}_clusters_{}_histogram.png'.format(OUTDIR, i, key))
            plt.savefig('{}/{}_clusters_{}_histogram.pdf'.format(OUTDIR, i, key))
            plt.savefig('{}/{}_clusters_{}_histogram.svg'.format(OUTDIR, i, key))
            plt.close()
