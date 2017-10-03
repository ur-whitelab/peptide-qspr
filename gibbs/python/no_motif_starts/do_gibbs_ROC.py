import numpy as np
from matplotlib import pyplot as plt
import sys

def printhelp():
    print("Usage: do_gpos_gibbs_ROC.py [gpos_peptides_file] [root_directory] [num_classes] [motif_length]")
    exit(1)

if len(sys.argv) != 5:
    printhelp()

INPUT = sys.argv[1]
DIRNAME = sys.argv[2]
NUM_MOTIF_CLASSES = int(sys.argv[3])
MOTIF_LENGTH = int(sys.argv[4])

DATA_DIR = '/home/rainier/pymc3_qspr/data/'
fakefile = DATA_DIR + 'pdb_distributed_apd_length_peps.txt'
fake_data = []
with open(fakefile, 'r') as datafile:
    fake_data=datafile.readlines()
for line in fake_data:
    line = line.replace('\n', '')


ALPHABET = ['A','R','N','D','C','Q','E','G','H','I',
            'L','K','M','F','P','S','T','W','Y','V']
def pep_to_int_list(pep):
    '''takes a single string of amino acids and translates to a list of ints'''
    return(list(map(ALPHABET.index, pep.replace('\n', ''))))

def read_data(datafile):
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
                        if(i < motif_start or i >= (motif_start + MOTIF_LENGTH)):#\geq because of indexing
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
                            prob += motif_dists[motif_class][i - j][peptide[i]] * start_dist[j] * class_dist[motif_class]
        else:#don't know class value OR motif start value. iterate through both...
            for i in range(length):
                for j in range(length - MOTIF_LENGTH+1):
                    for k in range(NUM_MOTIF_CLASSES):
                        if(i < j or i >= j+MOTIF_LENGTH):#not in a motif 
                            prob += bg_dist[peptide[i]] * start_dist[j] * class_dist[k]
                        else:#we are in a motif 
                            prob += motif_dists[k][i-j][peptide[i]] * start_dist[j]* class_dist[k]
    return(prob)


apd_data, all_apd_aa = read_data(INPUT)

keys = apd_data.keys()
#need to rebuild the distros from files
motif_dists = np.ones((NUM_MOTIF_CLASSES, MOTIF_LENGTH, len(ALPHABET))) / float(len(ALPHABET))
motif_start_dists = {}
motif_class_dists = {}


for key in keys:
    motif_start_dists[key] = np.ones((len(apd_data[key]), (key - MOTIF_LENGTH+1)))/float(key - MOTIF_LENGTH+1)
    motif_class_dists[key] = np.ones((len(apd_data[key]) , NUM_MOTIF_CLASSES)) / float(NUM_MOTIF_CLASSES)
    
for key in keys:#all keys
    for i in range(len(apd_data[key])):#all peptides
        motif_start_dists[key][i] = np.genfromtxt('{}/motif_length_{}_length_{}_start_dist.txt'.format(DIRNAME, MOTIF_LENGTH, key))
        motif_class_dists[key][i] = np.genfromtxt('{}/motif_length_{}_length_{}_index_{}_class_dist.txt'.format(DIRNAME, MOTIF_LENGTH, key, i))

for i in range(NUM_MOTIF_CLASSES):
    for j in range(MOTIF_LENGTH):
        motif_dists[i][j] = np.genfromtxt('{}/class_{}_of_{}_position_{}_motif_dist.txt'.format(DIRNAME,i,NUM_MOTIF_CLASSES, j))

bg_dist = np.genfromtxt('{}/bg_dist.txt'.format(DIRNAME))

#now that we've recovered the distros, time to DO THE ROC CALCULATIONS!


apd_probs_dict = {}
apd_probs_arr = []

for key in keys:
    for i in range(len(apd_data[key])):
        if(key in apd_probs_dict.keys()):
            apd_probs_dict[key].append(get_tot_prob(apd_data[key][i], bg_dist, motif_dists, motif_class_dists[key][i], motif_start_dists[key][i]))
        else:
            apd_probs_dict[key] = []
            apd_probs_dict[key].append(get_tot_prob(apd_data[key][i], bg_dist, motif_dists, motif_class_dists[key][i], motif_start_dists[key][i]))
        apd_probs_arr.append(get_tot_prob(apd_data[key][i], bg_dist, motif_dists, motif_class_dists[key][i], motif_start_dists[key][i]))

apd_probs_arr = np.array(apd_probs_arr)

