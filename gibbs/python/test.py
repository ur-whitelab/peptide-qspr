import numpy as np
import re
import math
import matplotlib as mpl
from matplotlib import pyplot as plt
import pdb
import sys
import libgibbs as lg

def printhelp():
    print("Usage: test_gibbs.py [test_peptides_file] [num_classes] [num_iterations] [learning_rate]")
    exit(1)

if len(sys.argv) != 5:
    printhelp()

INPUT = sys.argv[1]
NUM_MOTIF_CLASSES = int(sys.argv[2]) #one class for which letter the motif starts with? why not try
NRUNS = int(sys.argv[3])
ETA = float(sys.argv[4])


#CONSTANTS
MOTIF_LENGTH = 4 #fixed motif lengths, for now

ALPHABET = ['A','R','N','D','C','Q','E','G','H','I',
            'L','K','M','F','P','S','T','W','Y','V']
DATA_DIR = '/home/rainier/pymc3_qspr/data/'
fakefile = DATA_DIR + 'pdb_distributed_apd_length_peps.txt'

def pep_to_int_list(pep):
    '''takes a single string of amino acids and translates to a list of ints'''
    return(list(map(ALPHABET.index, pep)))

def read_data(datafile, motif_file=None):
    '''Takes a properly-formatted peptide datafile (each line MUST start with a sequence)
       and reads it into a list.'''
    data = {}#dict keyed by peptide length containing the sequences
    with open(datafile, 'r') as f:
        big_aa_string = ''#for training the whole background distro
        lines = f.readlines()
        nlines = len(lines)
        start_idx = (1 if ('#' in lines[0] or 'sequence' in lines[0]) else 0)
        for line in lines[start_idx:]:#skip the header
            pep = line.split(',')[0]
            length = len(pep)
            big_aa_string+=pep
            if(length not in data.keys()):
                data[length] = [(pep_to_int_list(pep))]
            else:
                data[length].append((pep_to_int_list(pep)))
        big_aa_list = pep_to_int_list(big_aa_string)
    return(data, big_aa_list)


apd_data, all_apd_aa  = read_data(INPUT)#('/home/rainier/pymc3_qspr/gibbs/control_peptides.txt')

#initialize the OVERALL distributions as uniform
motif_dists = np.ones((NUM_MOTIF_CLASSES, MOTIF_LENGTH, len(ALPHABET))) / float(len(ALPHABET))

tot_motif_counts = {}#keep track of raw counts for EACH peptide separately
for key in apd_data.keys():
    tot_motif_counts[key] = np.zeros((NUM_MOTIF_CLASSES, MOTIF_LENGTH, len(ALPHABET)))

motif_counts = {}#this one is for local counting within the loop only.
for key in apd_data.keys():
    motif_counts[key] = np.zeros((NUM_MOTIF_CLASSES, MOTIF_LENGTH, len(ALPHABET)))


bg_dist = np.zeros(len(ALPHABET))
#distributions not tracked by peptide, just length.
motif_start_dists = {}
motif_class_dists = {}
#raw counts tracked PER peptide
motif_start_counts = {}
#motif_class_counts = {}
for key in apd_data.keys():
    motif_start_dists[key] = np.ones((len(apd_data[key]), (key - MOTIF_LENGTH+1)))/float(key - MOTIF_LENGTH+1)
    motif_start_counts[key] = np.zeros((len(apd_data[key]), (key - MOTIF_LENGTH +1)), dtype = int)
    motif_class_dists[key] = np.ones((len(apd_data[key]) , NUM_MOTIF_CLASSES)) / float(NUM_MOTIF_CLASSES)
#    motif_class_counts[key] = np.zeros((len(apd_data[key]) ,NUM_MOTIF_CLASSES))
bg_counts = np.zeros(len(ALPHABET), dtype=int)#times we see each AA as a b/g element
tot_bg_counts = np.zeros(len(ALPHABET), dtype=int)#times we see each AA as a b/g element

bg_count_list = bg_counts.tolist()
tot_bg_count_list = tot_bg_counts.tolist()

test_class = lg.Gibbs_Py(apd_data,
                         motif_counts,
                         motif_start_dists,
                         motif_class_dists,
                         bg_count_list,
                         tot_bg_count_list,
                         NRUNS)
