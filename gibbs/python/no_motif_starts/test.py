import numpy as np
import re
import math
import matplotlib as mpl
from matplotlib import pyplot as plt
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

def pep_to_int_list(pep):
    '''takes a single string of amino acids and translates to a list of ints'''
    return(list(map(ALPHABET.index, pep)))

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
