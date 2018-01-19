import numpy as np
from matplotlib import pyplot as plt
import sys
import math

def printhelp():
    print("Usage: count_motifs.py [root_directory] [num_classes] [motif_length]")
    exit(1)

if len(sys.argv) != 4:
    printhelp()

DIRNAME = sys.argv[1]
NUM_MOTIF_CLASSES = int(sys.argv[2])
MOTIF_LENGTH = int(sys.argv[3])
    
TRAINFILE = DIRNAME + 'train_set.txt'
TESTFILE = DIRNAME + 'test_set.txt'

test_lines = []
train_lines = []
with open('{}/test_set.txt'.format(DIRNAME), 'r') as f:
    test_lines = f.readlines()
with open('{}/train_set.txt'.format(DIRNAME), 'r') as f:
    train_lines = f.readlines()
test_lines = [line.replace(',\n','') for line in test_lines]
train_lines = [line.replace(',\n','') for line in train_lines]

bg_dist = np.genfromtxt('{}/bg_dist.txt'.format(DIRNAME))
motif_dists = np.ones((NUM_MOTIF_CLASSES, MOTIF_LENGTH, len(ALPHABET))) / float(len(ALPHABET))
for i in range(NUM_MOTIF_CLASSES):
    for j in range(MOTIF_LENGTH):
        motif_dists[i][j] = np.genfromtxt('{}/class_{}_of_{}_position_{}_motif_dist.txt'.format(DIRNAME,i,NUM_MOTIF_CLASSES, j))

        
def calc_prob(peptide, bg_dist,  motif_dists, motif_class=None):
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
                        if(motif_class is None):
                            prob += motif_dists[k][i-j][peptide[i]] * start_dist[j]
                        else:
                            prob += motif_dists[motif_class][i-j][peptide[i]] * start_dist[j]
    else:#impossible to have a motif of this length, all b/g
        prob = 0.0
        for i in range(length):
            prob += bg_dist[peptide[i]]
    prob /= float(length)
    return(prob)

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

test_data, train_data, all_apd_aa = read_data(TRAINFILE, TESTFILE)
predict_counts = [0 for i in range(NUM_MOTIF_CLASSES)]#number of times model predicts we see a motif
actual_counts = [0 for i in range(NUM_MOTIF_CLASSES)]#number of times we actually see that motif
probs_arr = np.zeros(NUM_MOTIF_CLASSES)
print('PROCESSING MOTIF DATA...')
for key in test_data.keys():
    for i in range(len(test_data[key])):
        for j in range(NUM_MOTIF_CLASSES):
            probs_arr[j] = calc_prob(test_data[key][i], bg_dist, motif_dists, j)
        predict_counts[np.argmax(probs_arr)] += 1
for key in train_data.keys():
    for i in range(len(train_data[key])):
        for j in range(NUM_MOTIF_CLASSES):
            probs_arr[j] = calc_prob(train_data[key][i], bg_dist, motif_dists, j)
        predict_counts[np.argmax(probs_arr)] += 1

print('COUNTING MOTIF OCCURRENCES...')
motifs = [[[] for j in range(MOTIF_LENGTH)] for i in range(NUM_MOTIF_CLASSES) ]
for i in range(NUM_MOTIF_CLASSES):
    for j in range(MOTIF_LENGTH):
        for k in range(len(ALPHABET)):
            if motif_dists[i][j][k] > 0.05:#cutoff at 5%
                motifs[i][j].append(k)
motif_lists = {}
indices = []
'''for i in range(NUM_MOTIF_CLASSES):
    maxlength = 1
    for j in range(MOTIF_LENGTH):
        maxlength = max(maxlength, len(motifs[i][j]))
        if(len(motifs[i][j]) > 1):
            indices.append([i,j, len(motifs[i][j])])
    if(i in motif_lists.keys()):
        motif_lists[i].append([motifs[i][j][0] for j in range(MOTIF_LENGTH)])
    else:
        motif_lists[i] = ([motifs[i][j][0] for j in range(MOTIF_LENGTH)])
'''
#hard-coded for 5, uh oh
count = 0
for motif in motifs:
    indices = [len(item) for item in motif]
    for i in range(indices[0]):
        for j in range(indices[1]):
            for k in range(indices[2]):
                for a in range(indices[3]):
                    if(count not in motif_lists.keys()):
                        motif_lists[count] = [''.join(ALPHABET[motif[0][i]]+ ALPHABET[motif[1][j]]+ ALPHABET[motif[2][k]]+ ALPHABET[motif[3][a]])]
                    else:
                        motif_lists[count].append(''.join(ALPHABET[motif[0][i]]+ ALPHABET[motif[1][j]]+ ALPHABET[motif[2][k]]+ ALPHABET[motif[3][a]]))
    count += 1


for line in test_lines:
    for key in motif_lists.keys():
        for j in range(len(motif_lists[key])):
            if(motif_lists[key][j] in line):
                actual_counts[int(key)] += 1
                print('FOUND "{}" in "{}"!'.format(motif_lists[key][j], line))

for line in train_lines:
    for key in motif_lists.keys():
        for j in range(len(motif_lists[key])):
            if(motif_lists[key][j] in line):
                actual_counts[int(key)] += 1
                print('FOUND "{}" in "{}"!'.format(motif_lists[key][j], line))

np.savetxt('{}/predicted_counts.txt'.format(DIRNAME), predict_counts,'%5.2f')
np.savetxt('{}/found_counts.txt'.format(DIRNAME), actual_counts,'%5.2f')
with open('{}/motif_lists.txt'.format(DIRNAME), 'w+') as f:
    count = 1
    for motif in motifs:
        f.write('MOTIF {}:\n'.format(count))
        for place in motif:
            if(len(place) == 1):
                f.write(ALPHABET[place[0]])
            else:
                f.write('[')
                for item in place:
                    f.write(ALPHABET[item])
                f.write(']')
        f.write('\n')
        count+=1
