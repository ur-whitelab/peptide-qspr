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

_, _, train_data, test_data, all_apd_aa = read_logs(TRAINFILE, TESTFILE)

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
            test_probs_dict[key].append(calc_prob(test_data[key][i], bg_dist, motif_dists, num_motif_classes=NUM_MOTIF_CLASSES, motif_length=MOTIF_LENGTH))
        else:
            test_probs_dict[key] = []
            test_probs_dict[key].append(calc_prob(test_data[key][i], bg_dist, motif_dists, num_motif_classes=NUM_MOTIF_CLASSES, motif_length=MOTIF_LENGTH))
        test_probs_arr.append(calc_prob(test_data[key][i], bg_dist, motif_dists, num_motif_classes=NUM_MOTIF_CLASSES, motif_length=MOTIF_LENGTH))

for key in train_keys:
    for i in range(len(train_data[key])):
        if(key in train_probs_dict.keys()):
            train_probs_dict[key].append(calc_prob(train_data[key][i], bg_dist, motif_dists, num_motif_classes=NUM_MOTIF_CLASSES, motif_length=MOTIF_LENGTH))
        else:
            train_probs_dict[key] = []
            train_probs_dict[key].append(calc_prob(train_data[key][i], bg_dist, motif_dists, num_motif_classes=NUM_MOTIF_CLASSES, motif_length=MOTIF_LENGTH))
        train_probs_arr.append(calc_prob(train_data[key][i], bg_dist, motif_dists, num_motif_classes=NUM_MOTIF_CLASSES, motif_length=MOTIF_LENGTH))

for item in fake_data:
    fake_probs_arr.append(calc_prob(pep_to_int_list(item), bg_dist, motif_dists, num_motif_classes=NUM_MOTIF_CLASSES, motif_length=MOTIF_LENGTH))

test_probs_arr = np.array(test_probs_arr)
train_probs_arr = np.array(train_probs_arr)
fake_probs_arr = np.array(fake_probs_arr)

train_min, test_min, fakeset_min = np.min(train_probs_arr), np.min(test_probs_arr), np.min(fake_probs_arr)
train_max, test_max, fakeset_max = np.max(train_probs_arr), np.max(test_probs_arr), np.max(fake_probs_arr)

roc_min = min(train_min, test_min, fakeset_min)
roc_max = max(train_max, test_max, fakeset_max)

print("PLOTTING ROC DATA...")

FPR_ARR, TPR_ARR, _, CUTOFF, BEST_IDX = gen_roc_data(NPOINTS, roc_min, roc_max, fakes=fake_probs_arr, trains=train_probs_arr, devs=test_probs_arr)
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



