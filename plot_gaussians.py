import numpy as np
from matplotlib import pyplot as plt
import sys

def printHelp():
    print("Usage: plot_gaussians.py [directory] [min_num_hills] [max_num_hills]")
    exit(1)

if len(sys.argv) != 4:
    printHelp()

dirname = sys.argv[1]
min_num_hills = int(sys.argv[2])
max_num_hills = int(sys.argv[3])

keys = ['nChargedGroups_observed', 'netCharge_observed', 'nNonPolarGroups_observed']#constant

for i in range(min_num_hills, max_num_hills):
    for key in keys:
        counts = np.genfromtxt('{}/{}_clusters_{}.counts'.format(dirname, i, key))
        bins =  np.genfromtxt('{}/{}_clusters_{}.bins'.format(dirname, i, key))
        plt.figure()
        plt.bar(bins[1:], counts)
        plt.savefig('{}/{}_clusters_{}.png'.format(dirname, i, key))
        plt.savefig('{}/{}_clusters_{}.pdf'.format(dirname, i, key))
        plt.savefig('{}/{}_clusters_{}.svg'.format(dirname, i, key))
        plt.close()
