import numpy as np
from matplotlib import pyplot as plt
import sys
from qspr_plots.qspr_plots import *

def printhelp():
    print("Usage: do_gpos_gibbs_ROC.py [gpos_peptides_file] [root_directory] [num_classes] [motif_length]")
    exit(1)

if len(sys.argv) != 5:
    printhelp()

INPUT = sys.argv[1]
DIRNAME = sys.argv[2]
NUM_MOTIF_CLASSES = int(sys.argv[3])
MOTIF_LENGTH = int(sys.argv[4])

DATA_DIR = '/home/rainier/pymc3_qspr/data/'
fakefile = DATA_DIR + 'pdb_distributed_apd_length_peps.txt'
fake_data = []
with open(fakefile, 'r') as datafile:
    fake_data=datafile.readlines()
for line in fake_data:
    line = line.replace('\n', '')


apd_data, all_apd_aa = read_data(INPUT)

keys = apd_data.keys()
#need to rebuild the distros from files
motif_dists = np.ones((NUM_MOTIF_CLASSES, MOTIF_LENGTH, len(ALPHABET))) / float(len(ALPHABET))
motif_start_dists = {}
motif_class_dists = {}


for key in keys:
    motif_start_dists[key] = np.ones((len(apd_data[key]), (key - MOTIF_LENGTH+1)))/float(key - MOTIF_LENGTH+1)
    motif_class_dists[key] = np.ones((len(apd_data[key]) , NUM_MOTIF_CLASSES)) / float(NUM_MOTIF_CLASSES)
    
for key in keys:#all keys
    for i in range(len(apd_data[key])):#all peptides
        motif_start_dists[key][i] = np.genfromtxt('{}/motif_length_{}_length_{}_start_dist.txt'.format(DIRNAME, MOTIF_LENGTH, key))
        motif_class_dists[key][i] = np.genfromtxt('{}/motif_length_{}_length_{}_index_{}_class_dist.txt'.format(DIRNAME, MOTIF_LENGTH, key, i))

for i in range(NUM_MOTIF_CLASSES):
    for j in range(MOTIF_LENGTH):
        motif_dists[i][j] = np.genfromtxt('{}/class_{}_of_{}_position_{}_motif_dist.txt'.format(DIRNAME,i,NUM_MOTIF_CLASSES, j))

bg_dist = np.genfromtxt('{}/bg_dist.txt'.format(DIRNAME))

#now that we've recovered the distros, time to DO THE ROC CALCULATIONS!


apd_probs_dict = {}
apd_probs_arr = []

for key in keys:
    for i in range(len(apd_data[key])):
        if(key in apd_probs_dict.keys()):
            apd_probs_dict[key].append(get_tot_prob(apd_data[key][i], bg_dist, motif_dists, motif_class_dists[key][i], motif_start_dists[key][i]))
        else:
            apd_probs_dict[key] = []
            apd_probs_dict[key].append(get_tot_prob(apd_data[key][i], bg_dist, motif_dists, motif_class_dists[key][i], motif_start_dists[key][i]))
        apd_probs_arr.append(get_tot_prob(apd_data[key][i], bg_dist, motif_dists, motif_class_dists[key][i], motif_start_dists[key][i]))

apd_probs_arr = np.array(apd_probs_arr)

