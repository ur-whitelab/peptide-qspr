import numpy as np
import re
import math
import matplotlib as mpl
from matplotlib import pyplot as plt
import pdb
import sys
import libgibbs as lg
import random
import time
import sys



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

def get_tot_prob(peptide, bg_dist,  motif_dists, class_dist, start_dist, motif_class=None, motif_start=None ):
    '''Takes in a single peptide as a LIST OF INTS, the background distro, the
       dict of motif distros, either the class distro OF THE SPECIFIC PEPTIDE
       or the set motif class, and either the motif start position or the start 
       distros. Returns the total probability
       density assigned to the sequence by the model. This is called during sampling AND
       during calculation of ROC data, so it has to handle taking in distros or set values for both
       motif start and motif class in any combination (4 cases).'''
    length = len(peptide)
    prob = 0.0
    #for use during Gibbs steps, when start position is given (sampled)
    if motif_start is not None:#use set start position
        if motif_class is not None:#use set class value
            for i in range(length):#loop over all AA
                for j in range(length - MOTIF_LENGTH + 1):
                    for k in range(NUM_MOTIF_CLASSES):
                        #we know where the motif is
                        if(i < motif_start or i >= motif_start + MOTIF_LENGTH):#\geq because of indexing
                            prob += bg_dist[peptide[i]] * start_dist[j] * class_dist[motif_class]
                        else:
                            prob += motif_dists[motif_class][ i - motif_start][peptide[i]] * start_dist[j] * class_dist[motif_class]
        else:#motif_class is None -> use distros
            for i in range(length):
                for j in range(length - MOTIF_LENGTH + 1):
                    for k in range(NUM_MOTIF_CLASSES):
                        if(i < motif_start or i >= motif_start + MOTIF_LENGTH):#not in a motif
                            prob += bg_dist[peptide[i]]* start_dist[j] * class_dist[k]
                        else:#in a motif
                            prob += motif_dists[k][i - motif_start][peptide[i]] * start_dist[j] * class_dist[k]
                            #for use during evaluation & finding ROC data
    else:#start_dist is not None -> use distros, no set value
        if motif_class is not None:#use set class value but draw from start position distro
            for i in range(length):
                for j in range(length - MOTIF_LENGTH+1):#all possible motif start positions
                    for k in range(NUM_MOTIF_CLASSES):
                        if( i < j or i >= j+MOTIF_LENGTH):#we're not in a motif
                            prob += bg_dist[peptide[i]] * start_dist[j] * class_dist[k]
                        else:#we're in a motif
                            prob += motif_dists[motif_class][i - j][peptide[i]] * start_dist[j] * class_dist[k]
        else:#don't know class value OR motif start value. iterate through both...
            for i in range(length):
                for j in range(length - MOTIF_LENGTH+1):
                    for k in range(NUM_MOTIF_CLASSES):
                        if(i < j or i >= j+MOTIF_LENGTH):#not in a motif 
                            prob += bg_dist[peptide[i]] * start_dist[j] * class_dist[k]
                        else:#we are in a motif 
                            prob += motif_dists[k][i-j][peptide[i]] * start_dist[j]* class_dist[k]
    return(prob)


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


bg_dist = np.ones(len(ALPHABET))/float(len(ALPHABET))
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

RNG_SEED = 985871062

test_class = lg.Gibbs_Py(apd_data,
                         motif_counts,
                         motif_start_dists,
                         motif_class_dists,
                         motif_dists,
                         bg_count_list,
                         tot_bg_count_list,
                         NRUNS,
                         MOTIF_LENGTH,
                         NUM_MOTIF_CLASSES,
                         RNG_SEED)

for i in range(100):
    choice = test_class.test_random_choice(10)
    assert(choice == "SUCCESS")

print("BEGINNING TESTS...")

print("TOTAL ITERATIONS: {}".format(len(apd_data[5])))
for i in range(len(apd_data[5])):
#    print("starting iteration {}".format(i))
    prob = get_tot_prob(apd_data[5][i], bg_dist, motif_dists, motif_class_dists[5][i], motif_start_dists[5][i])
    #test that get_tot_prob works on a clean slate
    assert(test_class.test_get_tot_prob(prob, i) == "TEST PASSED")
    #test that the RNG seed worked
    assert(test_class.test_rng(random.random()) == "TEST PASSED")
#    print('made it through iteration {}'.format(i))

#    choice = int(choice[-1])
#    assert(choice > -1 and choice < 10)

print("TESTING THE run() METHOD...")
test_class.run()
print("run() METHOD SUCCEEDED")

print("ALL TESTS PASSED")


print("TIMING METHODS")

niters = 10000
start_time = time.time()
for i in range(niters):
    prob = get_tot_prob(apd_data[5][0], bg_dist, motif_dists, motif_class_dists[5][0], motif_start_dists[5][0])
end_time = time.time()
python_time = (end_time - start_time)
start_time = time.time()
test_class.time_get_tot_prob(niters, 0)
end_time = time.time()
cpp_time = (end_time - start_time)

print("FOR {} ITERATIONS OF get_tot_prob(), PYTHON TIME WAS {} AND C++ TIME WAS {}".format(niters, python_time, cpp_time))
print("SPEEDUP: ~{:.4}x".format(python_time/cpp_time))
