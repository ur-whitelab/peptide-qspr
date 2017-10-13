import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
import math
import copy

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
GPOSFILE = DATA_DIR + 'APD_GPOS_SEQS.out'
GPOS_DATA = pd.read_csv(GPOSFILE)


ALPHABET = ['A','R','N','D','C','Q','E','G','H','I',
            'L','K','M','F','P','S','T','W','Y','V']

def get_hist_prob(bins, counts, value):
    '''takes in the bins and counts from a normalized numpy.histogram() and a value,
        returns the height of the bin that value falls into.'''
    idx = np.argmax(bins > value)
    if idx==0:
        return(0)
    else:
        return(counts[idx-1]/np.sum(counts))
    
def pep_to_int_list(pep):
    '''takes a single string of amino acids and translates to a list of ints'''
    return(list(map(ALPHABET.index, pep.replace('\n', ''))))

def read_data(trainfile, testfile):
    '''Takes a properly-formatted peptide datafile (each line MUST start with a sequence)
       and reads it into a list.'''
    train_data = {}#dict keyed by peptide length containing the sequences
    test_data = {}
    test_peptides = []
    train_peptides = []
    big_aa_string = ''#for training the whole background distro
    with open(trainfile, 'r') as f:
        lines = f.readlines()
        nlines = len(lines)
        start_idx = (1 if ('#' in lines[0] or 'sequence' in lines[0]) else 0)
        for line in lines[start_idx:]:#skip the header
            pep = line.split(',')[0]
            train_peptides.append(pep)
            length = len(pep)
            big_aa_string+=pep
            if(length not in train_data.keys()):
                train_data[length] = [(pep_to_int_list(pep))]
            else:
                train_data[length].append((pep_to_int_list(pep)))
    with open(testfile, 'r') as f:
        lines = f.readlines()
        nlines = len(lines)
        start_idx = (1 if ('#' in lines[0] or 'sequence' in lines[0]) else 0)
        for line in lines[start_idx:]:#skip the header
            pep = line.split(',')[0]
            test_peptides.append(pep)
            length = len(pep)
            big_aa_string+=pep
            if(length not in test_data.keys()):
                test_data[length] = [(pep_to_int_list(pep))]
            else:
                test_data[length].append((pep_to_int_list(pep)))
    big_aa_list = pep_to_int_list(big_aa_string)
    return(test_peptides, train_peptides, train_data, test_data, big_aa_list)

def calc_prob(peptide, bg_dist,  motif_dists):
    '''For use when we're OUTSIDE the model, for generating ROC data and the like.'''
    length = len(peptide)
    if(length - MOTIF_LENGTH +1 > 0 and MOTIF_LENGTH > 0):
        start_dist = np.ones(length - MOTIF_LENGTH +1) /(length-MOTIF_LENGTH+1)#uniform start dists
        prob = 0.0
        for i in range(length):
            for j in range(length - MOTIF_LENGTH+1):
                for k in range(NUM_MOTIF_CLASSES):
                    if(i < j or i >= j+MOTIF_LENGTH):#not in a motif 
                        prob += bg_dist[peptide[i]] * start_dist[j]
                    else:#we are in a motif 
                        prob += motif_dists[k][i-j][peptide[i]] * start_dist[j]
    else:#impossible to have a motif of this length, all b/g
        prob = 0.0
        for i in range(length):
            prob += bg_dist[peptide[i]]
    prob /= float(length)
    return(prob)

def calc_positives(arr, cutoff):
    '''takes in an array of probs given by the above model and returns the number of
       probs above the cutoff probability. This is for use in generating the ROC curve.'''
    arr = np.sort(np.array(arr))
    if not arr[-1] < cutoff:
        return(len(arr) - np.argmax(arr > cutoff))
    else:
        return(0)

def gen_roc_data(npoints, fpr_arr, tpr_arr, roc_min, roc_max, fakes,
                 trains, tests):
    '''This fills two numpy arrays for use in plotting the ROC curve. The first is the FPR,
       the second is the TPR. The number of points is npoints. Returns (FPR_arr, TPR_arr).'''
    best_cutoff = 0.0
    best_ROC = 0.0
    roc_range = np.linspace(roc_min, roc_max, npoints)
    #for each cutoff value, calculate the FPR and TPR
    for i in range(npoints):
        fakeset_positives = calc_positives(fakes, roc_range[i])
        fpr_arr[i] = float(fakeset_positives) / len(fakes)
        test_positives =  calc_positives(tests, roc_range[i])
        train_positives = calc_positives(trains, roc_range[i])
        tpr_arr[i] = float(train_positives + test_positives) / (len(trains) + len(tests) )
    best_idx = 0
    old_dist = 2.0
    for i in range(0,npoints):
        dist = math.sqrt(PREFACTOR * fpr_arr[i] **2 + (1-tpr_arr[i]) **2)
        if (old_dist > dist and fpr_arr[i] > 0):
            best_idx = i
            old_dist = dist
    best_cutoff = roc_range[best_idx]
    accuracy = (tpr_arr[best_idx] + (1.0-fpr_arr[best_idx]))/2.0
    return( ( accuracy, best_idx, best_cutoff))

    
#The Gibbs part

print("READING DATA...")
test_peps, train_peps, test_seqs, train_seqs, all_apd_aa = read_data(TRAINFILE, TESTFILE)

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
        prob += get_hist_prob(bins[key], counts[key], GPOS_DATA.loc[GPOS_DATA['sequence'] == pep][key].iloc[0])/3.0
    test_gibbs_probs.append(calc_prob(pep_to_int_list(pep), bg_dist, motif_dists))
    test_gauss_probs.append(prob)

for pep in train_peps:
    prob = 0.0
    for key in keys:
        prob += get_hist_prob(bins[key], counts[key], GPOS_DATA.loc[GPOS_DATA['sequence'] == pep][key].iloc[0])/3.0
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

test_gibbs_probs = np.array(test_gibbs_probs)
train_gibbs_probs = np.array(train_gibbs_probs)
fake_gibbs_probs = np.array(fake_gibbs_probs)
test_gauss_probs = np.array(test_gauss_probs)
train_gauss_probs = np.array(train_gauss_probs)
fake_gauss_probs = np.array(fake_gauss_probs)


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
    accuracy, best_idx, best_cutoff = gen_roc_data(NPOINTS, fpr_arr, tpr_arr, roc_min, roc_max, roc_fake_probs, roc_train_probs, roc_test_probs)
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
plt.savefig('{}/{}_clusters_{}_motifs_length_{}_combined_statistics.svg'.format(DATA_DIR,NUM_CLUSTERS, NUM_MOTIF_CLASSES, MOTIF_LENGTH))
plt.savefig('{}/{}_clusters_{}_motifs_length_{}_combined_statistics.pdf'.format(DATA_DIR,NUM_CLUSTERS, NUM_MOTIF_CLASSES, MOTIF_LENGTH))
plt.savefig('{}/{}_clusters_{}_motifs_length_{}_combined_statistics.png'.format(DATA_DIR,NUM_CLUSTERS, NUM_MOTIF_CLASSES, MOTIF_LENGTH))


plt.figure()
plt.title('Optimal ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(optimal_fpr_arr[:-2], optimal_tpr_arr[:-2], 'o', color='red',label='ROC at varied cutoffs')
plt.plot(optimal_fpr_arr[optimal_best_idx], optimal_tpr_arr[optimal_best_idx], 's', color='blue', label='Optimal Cutoff')
plt.plot(optimal_fpr_arr,optimal_fpr_arr, color='black', ls=':', label='Totally Random')
plt.savefig('{}/{}_clusters_{}_motifs_length_{}_optimal_ROC_weight_{}.svg'.format(DATA_DIR,NUM_CLUSTERS, NUM_MOTIF_CLASSES, MOTIF_LENGTH, optimal_weight))
plt.savefig('{}/{}_clusters_{}_motifs_length_{}_optimal_ROC_weight_{}.png'.format(DATA_DIR,NUM_CLUSTERS, NUM_MOTIF_CLASSES, MOTIF_LENGTH, optimal_weight))
plt.savefig('{}/{}_clusters_{}_motifs_length_{}_optimal_ROC_weight_{}.pdf'.format(DATA_DIR,NUM_CLUSTERS, NUM_MOTIF_CLASSES, MOTIF_LENGTH, optimal_weight))


'''roc_min = min(min(test_gibbs_probs), min(train_gibbs_probs), min(fake_gibbs_probs))
roc_max = max(max(test_gibbs_probs), max(train_gibbs_probs), max(fake_gibbs_probs))
gibbs_acc, gibbs_best_idx, gibbs_best_cutoff = gen_roc_data(NPOINTS, fpr_arr, tpr_arr, roc_min, roc_max, fake_gibbs_probs, train_gibbs_probs, test_gibbs_probs)
roc_min = min(min(test_gauss_probs), min(train_gauss_probs), min(fake_gauss_probs))
roc_max = max(max(test_gauss_probs), max(train_gauss_probs), max(fake_gauss_probs))
gauss_acc, gauss_best_idx, gauss_best_cutoff = gen_roc_data(NPOINTS, fpr_arr, tpr_arr, roc_min, roc_max, fake_gauss_probs, train_gauss_probs, test_gauss_probs)

def alt_gen_roc_data(gauss_cutoff, gibbs_cutoff, gibbs_tests, gibbs_trains, gibbs_fakes, gauss_tests, gauss_trains, gauss_fakes):
    Only return a positive if BOTH Gibbs and Gauss models have it above the cutoff...
    true_positives = 0
    false_positives = 0
    count = 0
    for i in range(len(gibbs_tests)):
        count += 1
        if(gibbs_tests[i] > gibbs_cutoff and gauss_tests[i] > gauss_cutoff):
            true_positives += 1#only return positive if BOTH are positive
    for i in range(len(gibbs_trains)):
        count += 1
        if(gibbs_trains[i] > gibbs_cutoff and gauss_trains[i] > gauss_cutoff):
            true_positives += 1
    for i in range(len(gibbs_fakes)):
        count += 1
        if(gibbs_fakes[i] > gibbs_cutoff and gauss_fakes[i] > gauss_cutoff):
            false_positives += 1
    overall_tpr = float(true_positives)/float(count)
    overall_fpr = float(false_positives)/float(count)
    overall_accuracy = (overall_tpr + (1.0 - overall_fpr))/2.0
    
    return( ( overall_accuracy, overall_fpr, overall_tpr))

exp_acc, exp_fpr, exp_tpr = alt_gen_roc_data(gauss_best_cutoff, gibbs_best_cutoff, test_gibbs_probs, train_gibbs_probs, fake_gibbs_probs, test_gauss_probs, train_gauss_probs, fake_gauss_probs)

print("EXPERIMENTAL ACCURACY: {}".format(exp_acc))
print("EXPERIMENTAL FPR: {}".format(exp_fpr))
print("EXPERIMENTAL TPR: {}".format(exp_tpr))
'''
