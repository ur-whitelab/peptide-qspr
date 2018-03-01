import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
import math
import copy
from qspr_plots import *

def printhelp():
    print("Usage: do_combined_ROC.py [gaussmix_directory] [num_gauss_clusters] [gauss_ROC_distance_weight] [gibbs_directory] [num_motif_classes] [motif_length]")
    exit(1)

if len(sys.argv) != 7:
    printhelp()

GAUSSDIR = sys.argv[1]
NUM_CLUSTERS = int(sys.argv[2])
PREFACTOR = float(sys.argv[3])
GIBBSDIR = sys.argv[4]
NUM_MOTIF_CLASSES = int(sys.argv[5])
MOTIF_LENGTH = int(sys.argv[6])
DATA_DIR = '/home/rainier/pymc3_qspr/data/'
TRAINFILE = GIBBSDIR+'/train_set.txt'
TESTFILE = GIBBSDIR+'/test_set.txt'
FAKEFILE = DATA_DIR + 'shorter_pdb_distributed_peps.out'
FAKE_DATA = pd.read_csv(FAKEFILE)
HUMANFILE = DATA_DIR + 'Human_all.out'
HUMAN_DATA = pd.read_csv(HUMANFILE)


#The Gibbs part

print("READING DATA...")
train_peps, test_peps, train_seqs, test_seqs, all_apd_aa = read_logs(TRAINFILE, TESTFILE)

motif_dists = np.ones((NUM_MOTIF_CLASSES, MOTIF_LENGTH, len(ALPHABET))) / float(len(ALPHABET))

for i in range(NUM_MOTIF_CLASSES):
    for j in range(MOTIF_LENGTH):
        motif_dists[i][j] = np.genfromtxt('{}/class_{}_of_{}_position_{}_motif_dist.txt'.format(GIBBSDIR,i,NUM_MOTIF_CLASSES, j))

bg_dist = np.genfromtxt('{}/bg_dist.txt'.format(GIBBSDIR))



#the Gaussmix part
keys = ['netCharge', 'nChargedGroups', 'nNonPolarGroups']#the 3 key descriptors
counts = {}
bins = {}
for key in keys:
    counts[key] = np.genfromtxt('{}/{}_clusters_{}_observed.counts'.format(GAUSSDIR, NUM_CLUSTERS, key))
    bins[key] = np.genfromtxt('{}/{}_clusters_{}_observed.bins'.format(GAUSSDIR, NUM_CLUSTERS, key))


print("CALCULATING PROBABILITIES...")
#real data
test_gauss_probs = []
test_gibbs_probs = []
train_gauss_probs = []
train_gibbs_probs = []
#fake data
fake_gauss_probs = []
fake_gibbs_probs = []

for pep in test_peps:
    prob = 0.0
    for key in keys:
        prob += get_hist_prob(bins[key], counts[key], HUMAN_DATA.loc[HUMAN_DATA['sequence'] == pep][key].iloc[0])/3.0
    test_gibbs_probs.append(calc_prob(pep_to_int_list(pep), bg_dist, motif_dists))
    test_gauss_probs.append(prob)

for pep in train_peps:
    prob = 0.0
    for key in keys:
        prob += get_hist_prob(bins[key], counts[key], HUMAN_DATA.loc[HUMAN_DATA['sequence'] == pep][key].iloc[0])/3.0
    train_gibbs_probs.append(calc_prob(pep_to_int_list(pep), bg_dist, motif_dists))
    train_gauss_probs.append(prob)

for i in range(len(FAKE_DATA['sequence'])):
    prob = 0.0
    pep = FAKE_DATA['sequence'].iloc[i]
    for key in keys:
        prob += get_hist_prob(bins[key], counts[key], FAKE_DATA[key].iloc[i])/3.0
    fake_gibbs_probs.append(calc_prob(pep_to_int_list(pep), bg_dist, motif_dists))
    fake_gauss_probs.append(prob)

#divide each prob arr by the most likely prob to compare
#lowest_gibbs =  min( min(test_gibbs_probs), min(train_gibbs_probs), min(fake_gibbs_probs) )
#lowest_gauss = min( min(test_gauss_probs), min(train_gauss_probs), min(fake_gauss_probs) )
#test_gibbs_probs -= lowest_gibbs
#train_gibbs_probs -= lowest_gibbs
#fake_gibbs_probs -= lowest_gibbs
#test_gauss_probs -= lowest_gauss
#train_gauss_probs -= lowest_gauss
#fake_gauss_probs -= lowest_gauss
biggest_gibbs = max( max(test_gibbs_probs), max(train_gibbs_probs), max(fake_gibbs_probs) )
biggest_gauss = max( max(test_gauss_probs), max(train_gauss_probs), max(fake_gauss_probs) )

test_gibbs_probs /= biggest_gibbs
train_gibbs_probs /= biggest_gibbs
fake_gibbs_probs /= biggest_gibbs
test_gauss_probs /= biggest_gauss
train_gauss_probs /= biggest_gauss
fake_gauss_probs /= biggest_gauss

#test_gibbs_probs = np.array(test_gibbs_probs)
#train_gibbs_probs = np.array(train_gibbs_probs)
#fake_gibbs_probs = np.array(fake_gibbs_probs)
#test_gauss_probs = np.array(test_gauss_probs)
#train_gauss_probs = np.array(train_gauss_probs)
#fake_gauss_probs = np.array(fake_gauss_probs)


'''Now that the prob arrays are comparable magnitudes, we iterate through weights from 0.0 to 1.0
assigned to either one, and get our ROC for each weight, then we see which weighting is best and record that accuracy'''

NPOINTS = 1000#lots of dots
weights = np.linspace(0.0, 1.0, 101)
#these are re-used for each weight
roc_fake_probs = np.zeros(len(fake_gibbs_probs))
roc_train_probs = np.zeros(len(train_gibbs_probs))
roc_test_probs = np.zeros(len(test_gibbs_probs))
tpr_arr = np.zeros(NPOINTS)
fpr_arr = np.zeros(NPOINTS)
accuracy = 0.0
best_idx = 0
#these track the best statistics we get with each weight
best_fprs_arr = np.zeros(len(weights))
best_tprs_arr = np.zeros(len(weights))
best_accs_arr = np.zeros(len(weights))
optimal_fpr_arr = np.zeros(NPOINTS)
optimal_tpr_arr = np.zeros(NPOINTS)
optimal_best_idx = 0
optimal_acc = 0.0
optimal_weight = -1

print("CALCULATING ROC DATA, FINDING BEST WEIGHTING...")

for i in range(len(weights)):
    roc_fake_probs = weights[i] * fake_gibbs_probs + (1.0 - weights[i]) * fake_gauss_probs
    roc_train_probs = weights[i] * train_gibbs_probs + (1.0 - weights[i]) * train_gauss_probs
    roc_test_probs = weights[i] * test_gibbs_probs + (1.0 - weights[i]) * test_gauss_probs
    roc_min = min(np.min(roc_train_probs), np.min(roc_test_probs), np.min(roc_fake_probs))
    roc_max = max(np.max(roc_train_probs), np.max(roc_test_probs), np.max(roc_fake_probs))
    fpr_arr, tpr_arr, accuracy, best_cutoff, best_idx = gen_roc_data(NPOINTS, roc_min, roc_max, fakes=roc_fake_probs, trains=roc_train_probs, devs=roc_test_probs)
    best_fprs_arr[i] = fpr_arr[best_idx]
    best_tprs_arr[i] = tpr_arr[best_idx]
    best_accs_arr[i] = accuracy
    if(accuracy >= optimal_acc and (weights[i] != 0 and weights[i] != 1.0)):
        optimal_acc = accuracy
        optimal_fpr_arr = copy.deepcopy(fpr_arr)
        optimal_tpr_arr = copy.deepcopy(tpr_arr)
        optimal_best_idx = best_idx
        optimal_weight = weights[i]
    
    
plt.figure()
plt.title('Statistics as Weighting Varies')
plt.xlabel('Weight Assigned to Motifs')
plt.ylabel('Fraction')
plt.grid(color='grey', linestyle='--')
plt.plot(weights, best_fprs_arr, 'o', color = 'red', ls='--', label='FPR')
plt.plot(weights, best_tprs_arr, 'o', color = 'green', ls='--',label='TPR')
plt.plot(weights, best_accs_arr, 'o', color='blue', ls='--',label='Accuracy')
plt.legend(loc='best')
plt.savefig('{}/HUMAN_{}_clusters_{}_motifs_length_{}_combined_statistics.svg'.format(DATA_DIR,NUM_CLUSTERS, NUM_MOTIF_CLASSES, MOTIF_LENGTH))
plt.savefig('{}/HUMAN_{}_clusters_{}_motifs_length_{}_combined_statistics.pdf'.format(DATA_DIR,NUM_CLUSTERS, NUM_MOTIF_CLASSES, MOTIF_LENGTH))
plt.savefig('{}/HUMAN_{}_clusters_{}_motifs_length_{}_combined_statistics.png'.format(DATA_DIR,NUM_CLUSTERS, NUM_MOTIF_CLASSES, MOTIF_LENGTH))


plt.figure()
plt.title('Optimal ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(optimal_fpr_arr, optimal_tpr_arr, 'o', color='red',label='ROC at varied cutoffs')
plt.plot(optimal_fpr_arr[optimal_best_idx], optimal_tpr_arr[optimal_best_idx], 's', color='blue', label='Optimal Cutoff')
plt.plot(optimal_fpr_arr,optimal_fpr_arr, color='black', ls=':', label='Totally Random')
plt.savefig('{}/HUMAN_{}_clusters_{}_motifs_length_{}_optimal_ROC_weight_{}.svg'.format(DATA_DIR,NUM_CLUSTERS, NUM_MOTIF_CLASSES, MOTIF_LENGTH, optimal_weight))
plt.savefig('{}/HUMAN_{}_clusters_{}_motifs_length_{}_optimal_ROC_weight_{}.png'.format(DATA_DIR,NUM_CLUSTERS, NUM_MOTIF_CLASSES, MOTIF_LENGTH, optimal_weight))
plt.savefig('{}/HUMAN_{}_clusters_{}_motifs_length_{}_optimal_ROC_weight_{}.pdf'.format(DATA_DIR,NUM_CLUSTERS, NUM_MOTIF_CLASSES, MOTIF_LENGTH, optimal_weight))
