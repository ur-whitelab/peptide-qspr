import numpy as np
import sys
from matplotlib import pyplot as plt
import regex as re
from qspr_plots import *

def printhelp():
    print("Usage: align_seqs.py [root_directory] [num_classes] [motif_length]")
    exit(1)

if len(sys.argv) != 4:
    printhelp()

DIRNAME = sys.argv[1]
NUM_MOTIF_CLASSES = int(sys.argv[2])
MOTIF_LENGTH = int(sys.argv[3])

DATA_DIR = '/home/rainier/pymc3_qspr/data/'
TRAINFILE = DIRNAME+'/train_set.txt'
TESTFILE = DIRNAME+'/test_set.txt'
fakefile = DATA_DIR + 'pdb_distributed_apd_length_peps.txt'
fake_data = []
with open(fakefile, 'r') as datafile:
    fake_data=datafile.readlines()
for i in range(len(fake_data)) :
    fake_data[i] = fake_data[i].replace('\n', '')


def int_list_to_pep(int_list):
    '''takes a list of AA indices and returns the string it corresponds to'''
    ret = ''
    for item in int_list:
        ret += (ALPHABET[item])
    return(ret)

_, _, test_data, train_data, all_apd_aa, all_apd_strings = read_logs(TRAINFILE, TESTFILE, return_strings=True)

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
test_starts_probs_dict = {}
train_starts_probs_dict = {}

for key in test_keys:
    for i in range(len(test_data[key])):
        if(key in test_starts_probs_dict.keys()):
            test_starts_probs_dict[key].append([(calc_prob(test_data[key][i], bg_dist, motif_dists, motif_start = j, num_motif_classes=NUM_MOTIF_CLASSES, motif_length=MOTIF_LENGTH)) for j in range(len(test_data[key][i]) - MOTIF_LENGTH+1)])
        else:
            test_starts_probs_dict[key] = [[(calc_prob(test_data[key][i], bg_dist, motif_dists, motif_start = j, num_motif_classes=NUM_MOTIF_CLASSES, motif_length=MOTIF_LENGTH)) for j in range(len(test_data[key][i]) - MOTIF_LENGTH+1)]]

for key in train_keys:
    for i in range(len(train_data[key])):
        if(key in train_starts_probs_dict.keys()):
            train_starts_probs_dict[key].append([(calc_prob(train_data[key][i], bg_dist, motif_dists, motif_start = j, num_motif_classes=NUM_MOTIF_CLASSES, motif_length=MOTIF_LENGTH)) for j in range(len(train_data[key][i]) - MOTIF_LENGTH+1)])
        else:
            train_starts_probs_dict[key] = [[(calc_prob(train_data[key][i], bg_dist, motif_dists, motif_start = j, num_motif_classes=NUM_MOTIF_CLASSES, motif_length=MOTIF_LENGTH)) for j in range(len(train_data[key][i]) - MOTIF_LENGTH+1)]]


#get the most probable start for each of these peptides.
test_starts_dict = {}
train_starts_dict = {}
for key in test_starts_probs_dict:
    test_starts_dict[key] = [np.argmax(item) for item in test_starts_probs_dict[key]]
    
for key in train_starts_probs_dict:
    train_starts_dict[key] = [np.argmax(item) for item in train_starts_probs_dict[key]]


#now we find the actual sequences that appear in each of those places.
test_seqs_dict = {}
train_seqs_dict = {}

for key in test_starts_dict:
    test_seqs_dict[key] = [test_data[key][i][test_starts_dict[key][i]:test_starts_dict[key][i]+MOTIF_LENGTH] for i in range(len(test_starts_dict[key]))]
for key in train_starts_dict:
    train_seqs_dict[key] = [train_data[key][i][train_starts_dict[key][i]:train_starts_dict[key][i]+MOTIF_LENGTH] for i in range(len(train_starts_dict[key]))]

for key in test_starts_dict:
    test_seqs_dict[key] = [int_list_to_pep(item) for item in test_seqs_dict[key]]
for key in train_starts_dict:
    train_seqs_dict[key] = [int_list_to_pep(item) for item in train_seqs_dict[key]]

#now see if the motifs match up

with open(DIRNAME + '/motif_lists.txt', 'r') as f:
    lines = f.readlines()
lines = [item.replace('\n','') for item in lines]
predicted_motif_strings = [lines[2*i+1] for i in range(NUM_MOTIF_CLASSES)]
counts = [0 for item in predicted_motif_strings]

for i in range(len( predicted_motif_strings)):
    for key in test_seqs_dict:
        for item in test_seqs_dict[key]:
            if(len(re.findall(predicted_motif_strings[i], item)) > 0):
                counts[i] += len(re.findall(predicted_motif_strings[i], item))
    for key in train_seqs_dict:
        for item in train_seqs_dict[key]:
            if(len(re.findall(predicted_motif_strings[i], item)) > 0):
                counts[i] += len(re.findall(predicted_motif_strings[i], item))
#need to process the counts now, etc.
