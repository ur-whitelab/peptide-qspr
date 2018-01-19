import numpy as np
from matplotlib import pyplot as plt
import sys
import math
from qspr_plots import *

def printhelp():
    print("Usage: do_gibbs_ROC.py [root_directory] [num_classes] [motif_length]")
    exit(1)

if len(sys.argv) != 4:
    printhelp()

DIRNAME = sys.argv[1]
NUM_MOTIF_CLASSES = int(sys.argv[2])
MOTIF_LENGTH = int(sys.argv[3])

NPOINTS = 5000

DATA_DIR = '/home/rainier/pymc3_qspr/data/'
TRAINFILE = DIRNAME+'/train_set.txt'
TESTFILE = DIRNAME+'/test_set.txt'
fakefile = DATA_DIR + 'pdb_distributed_apd_length_peps.txt'
fake_data = []
with open(fakefile, 'r') as datafile:
    fake_data=datafile.readlines()
for i in range(len(fake_data)) :
    fake_data[i] = fake_data[i].replace('\n', '')


def read_data(trainfile, testfile):
    '''Takes a properly-formatted peptide datafile (each line MUST start with a sequence)
       and reads it into a list.'''
    train_data = {}#dict keyed by peptide length containing the sequences
    test_data = {}
    big_aa_string = ''#for training the whole background distro
    with open(trainfile, 'r') as f:
        lines = f.readlines()
        nlines = len(lines)
        start_idx = (1 if ('#' in lines[0] or 'sequence' in lines[0]) else 0)
        for line in lines[start_idx:]:#skip the header
            pep = line.split(',')[0]
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
            length = len(pep)
            big_aa_string+=pep
            if(length not in test_data.keys()):
                test_data[length] = [(pep_to_int_list(pep))]
            else:
                test_data[length].append((pep_to_int_list(pep)))
    big_aa_list = pep_to_int_list(big_aa_string)
    return(train_data, test_data, big_aa_list)

def calc_prob(peptide, bg_dist,  motif_dists):
    '''For use when we're OUTSIDE the model, for geinerating ROC data and the like.'''
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
    
def gen_roc_data(npoints, roc_min, roc_max, fakes,
                 trains, tests, fpr_arr, tpr_arr):
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
    for i in range(npoints):
        dist = math.sqrt(2.0 * fpr_arr[i] **2 + (1-tpr_arr[i]) **2)
        if (old_dist > dist and not (i==0 or i==NPOINTS-1)):
            best_idx = i
            old_dist = dist
    best_cutoff = roc_range[best_idx]
    print('best index was {}'.format(best_idx))
    return( (best_cutoff, best_idx))


test_data, train_data, all_apd_aa = read_data(TRAINFILE, TESTFILE)

test_keys = test_data.keys()
train_keys = train_data.keys()
#need to rebuild the distros from files
motif_dists = np.ones((NUM_MOTIF_CLASSES, MOTIF_LENGTH, len(ALPHABET))) / float(len(ALPHABET))

for i in range(NUM_MOTIF_CLASSES):
    for j in range(MOTIF_LENGTH):
        motif_dists[i][j] = np.genfromtxt('{}/class_{}_of_{}_position_{}_motif_dist.txt'.format(DIRNAME,i,NUM_MOTIF_CLASSES, j))

bg_dist = np.genfromtxt('{}/bg_dist.txt'.format(DIRNAME))

#now that we've recovered the distros, time to DO THE ROC CALCULATIONS!

print("NUMBER OF MOTIFS: {}".format(NUM_MOTIF_CLASSES))
print("MOTIF LENGTH: {}".format(MOTIF_LENGTH))

print("CALCULATING PROBABILITIES...")
test_probs_dict = {}
test_probs_arr = []
train_probs_dict = {}
train_probs_arr = []
fake_probs_arr = []
for key in test_keys:
    for i in range(len(test_data[key])):
        if(key in test_probs_dict.keys()):
            test_probs_dict[key].append(calc_prob(test_data[key][i], bg_dist, motif_dists))
        else:
            test_probs_dict[key] = []
            test_probs_dict[key].append(calc_prob(test_data[key][i], bg_dist, motif_dists))
        test_probs_arr.append(calc_prob(test_data[key][i], bg_dist, motif_dists))

for key in train_keys:
    for i in range(len(train_data[key])):
        if(key in train_probs_dict.keys()):
            train_probs_dict[key].append(calc_prob(train_data[key][i], bg_dist, motif_dists))
        else:
            train_probs_dict[key] = []
            train_probs_dict[key].append(calc_prob(train_data[key][i], bg_dist, motif_dists))
        train_probs_arr.append(calc_prob(train_data[key][i], bg_dist, motif_dists))

for item in fake_data:
    fake_probs_arr.append(calc_prob(pep_to_int_list(item), bg_dist, motif_dists))

test_probs_arr = np.array(test_probs_arr)
train_probs_arr = np.array(train_probs_arr)
fake_probs_arr = np.array(fake_probs_arr)

train_min, test_min, fakeset_min = np.min(train_probs_arr), np.min(test_probs_arr), np.min(fake_probs_arr)
train_max, test_max, fakeset_max = np.max(train_probs_arr), np.max(test_probs_arr), np.max(fake_probs_arr)

roc_min = min(train_min, test_min, fakeset_min)
roc_max = max(train_max, test_max, fakeset_max)

print("PLOTTING ROC DATA...")
FPR_ARR = np.zeros(NPOINTS)
TPR_ARR = np.zeros(NPOINTS)

CUTOFF, BEST_IDX = gen_roc_data(NPOINTS, roc_min, roc_max, fake_probs_arr, train_probs_arr, test_probs_arr, FPR_ARR, TPR_ARR)
print("best cutoff value: {}".format(CUTOFF))
print("using {} as our cutoff, we achieved an FPR of {} and a TPR of {}".format(CUTOFF, FPR_ARR[BEST_IDX], TPR_ARR[BEST_IDX]))

plt.rcParams.update({'font.size': 7})
plt.figure(figsize = (2.5, 2.0), dpi=800)
plt.xlabel('FPR')#, labelpad=-12)
plt.ylabel('TPR')#, labelpad=-18)
#plt.title('ROC Curve')
plt.plot(FPR_ARR[:-1], TPR_ARR[:-1], 'o', label='ROC at varied cutoffs', color='red', lw=2.0, ms=2.0)
plt.plot(FPR_ARR[BEST_IDX], TPR_ARR[BEST_IDX], 's', label='Best cutoff', color='blue', lw=2.0, ms=2.0)
plt.plot(FPR_ARR, FPR_ARR, label='totally random', ls=':', color='black', lw=2.0, ms=2.0)
plt.legend(loc='best', fontsize='small')
plt.tight_layout()
plt.savefig('{}/{}_motifs_length_{}_ROC.png'.format(DIRNAME, NUM_MOTIF_CLASSES, MOTIF_LENGTH))
plt.savefig('{}/{}_motifs_length_{}_ROC.svg'.format(DIRNAME, NUM_MOTIF_CLASSES, MOTIF_LENGTH))
plt.savefig('{}/{}_motifs_length_{}_ROC.pdf'.format(DIRNAME, NUM_MOTIF_CLASSES, MOTIF_LENGTH))

print("SAVING STATISTICS...")
best_tpr = TPR_ARR[BEST_IDX]
best_fpr = FPR_ARR[BEST_IDX]
accuracy = (best_tpr + (1.0-best_fpr))/2.0
with open('{}/{}_classes_length_{}_ROC_log.txt'.format(DIRNAME, NUM_MOTIF_CLASSES, MOTIF_LENGTH), 'w+') as f:
    f.write('best idx {}\n'.format(BEST_IDX))
    f.write('best FPR {}\n'.format(FPR_ARR[BEST_IDX]))
    f.write('best TPR {}\n'.format(TPR_ARR[BEST_IDX]))
    f.write('accuracy {}\n'.format(accuracy))



