import numpy as np
from matplotlib import pyplot as plt
import sys
import math
from qspr_plots import *

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

        
_, _, train_data, test_data, all_apd_aa = read_logs(TRAINFILE, TESTFILE)


predict_counts = [0 for i in range(NUM_MOTIF_CLASSES)]#number of times model predicts we see a motif
actual_counts = [0 for i in range(NUM_MOTIF_CLASSES)]#number of times we actually see that motif
probs_arr = np.zeros(NUM_MOTIF_CLASSES)
print('PROCESSING MOTIF DATA...')
for key in test_data.keys():
    for i in range(len(test_data[key])):
        for j in range(NUM_MOTIF_CLASSES):
            probs_arr[j] = calc_prob(test_data[key][i], bg_dist, motif_dists, motif_class=j, num_motif_classes=NUM_MOTIF_CLASSES, motif_length=MOTIF_LENGTH)
        predict_counts[np.argmax(probs_arr)] += 1
for key in train_data.keys():
    for i in range(len(train_data[key])):
        for j in range(NUM_MOTIF_CLASSES):
            probs_arr[j] = calc_prob(train_data[key][i], bg_dist, motif_dists, motif_class=j, num_motif_classes=NUM_MOTIF_CLASSES, motif_length=MOTIF_LENGTH)
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
                if(count not in motif_lists.keys()):
                    motif_lists[count] = [''.join(ALPHABET[motif[0][i]]+ ALPHABET[motif[1][j]]+ ALPHABET[motif[2][k]])]
                else:
                    motif_lists[count].append(''.join(ALPHABET[motif[0][i]]+ ALPHABET[motif[1][j]]+ ALPHABET[motif[2][k]]))
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
