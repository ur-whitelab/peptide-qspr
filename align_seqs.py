import numpy as np
from random import randint, shuffle
import sys
import pandas as pd
from sklearn import svm
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
import math

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


ALPHABET = ['A','R','N','D','C','Q','E','G','H','I',
            'L','K','M','F','P','S','T','W','Y','V']
def pep_to_int_list(pep):
    '''takes a single string of amino acids and translates to a list of ints'''
    return(list(map(ALPHABET.index, pep.replace('\n', ''))))

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

def calc_prob(peptide, bg_dist,  motif_dists, motif_start=None):
    '''For use when we're OUTSIDE the model, gives prob with the motif starting at a specified
       location.'''
    length = len(peptide)
    if(motif_start is None):
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
    else:
        if(length - MOTIF_LENGTH +1 > 0 and MOTIF_LENGTH > 0):
            start_dist = np.ones(length - MOTIF_LENGTH +1) /(length-MOTIF_LENGTH+1)#uniform start dists
            prob = 0.0
            for i in range(length):
                for j in range(length - MOTIF_LENGTH+1):
                    for k in range(NUM_MOTIF_CLASSES):
                        if(i < motif_start or i >= motif_start+MOTIF_LENGTH):#not in a motif 
                            prob += bg_dist[peptide[i]] * start_dist[motif_start]
                        else:#we are in a motif 
                            prob += motif_dists[k][i-motif_start][peptide[i]] * start_dist[motif_start]
        else:#impossible to have a motif of this length, all b/g
            prob = 0.0
            for i in range(length):
                prob += bg_dist[peptide[i]]
    prob /= float(length)
    return(prob)


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
starts_dict = {}

for key in test_keys:
    for i in range(len(test_data[key])):
        if(key in starts_dict.keys()):
            starts_dict[key].append([(calc_prob(test_data[key][i], bg_dist, motif_dists, motif_start = j)) for j in range(len(test_data[key][i]) - MOTIF_LENGTH+1)])
        else:
            starts_dict[key] = [[(calc_prob(test_data[key][i], bg_dist, motif_dists, motif_start = j)) for j in range(len(test_data[key][i]) - MOTIF_LENGTH+1)]]

for key in train_keys:
    for i in range(len(train_data[key])):
        if(key in starts_dict.keys()):
            starts_dict[key].append([(calc_prob(train_data[key][i], bg_dist, motif_dists, motif_start = j)) for j in range(len(train_data[key][i]) - MOTIF_LENGTH+1)])
        else:
            starts_dict[key] = [[(calc_prob(train_data[key][i], bg_dist, motif_dists, motif_start = j)) for j in range(len(train_data[key][i]) - MOTIF_LENGTH+1)]]

        
#time to plot stuff!






