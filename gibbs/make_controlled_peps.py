import numpy as np
from qspr_plots import *

CONTROL_MOTIF_POS = 1 #always start at second place...
MOTIF_LENGTH = 4 #fixed motif lengths, for now

def read_dists(aa_dist_file, length_file):
    '''Reads a distro file such as one created by prepare_pdb.py and returns the distros.'''
    aa_dist = np.genfromtxt(aa_dist_file)
    with open(length_file, 'r') as f:
        lengths = [int(line) for line in f]
    return(aa_dist, lengths)

def make_controlled_pep(bg_dist, motif_dist, motif_start, length):
    '''Takes in some defined distros, a motif position (fixed) and a length. Makes a peptide from
    them. These will be used to test the convergence of the Gibbs sampler.'''
    pep = []
    for i in range(length):
        if( i < motif_start or i >= motif_start+MOTIF_LENGTH):#not in motif
            aa_idx = np.random.choice(range(len(ALPHABET)), p=bg_dist)
            pep+=(ALPHABET[aa_idx])
        else:
            aa_idx = np.random.choice(range(len(ALPHABET)), p=motif_dist[i-motif_start])
            pep+=(ALPHABET[aa_idx])
    pep+=',\n'
    return(pep)

apd_data, _ = read_data('/home/rainier/pymc3_qspr/data/APD_qsar_new.txt')

control_bg_dist = np.ones(len(ALPHABET)) / len(ALPHABET)
control_motif_dist = np.zeros((MOTIF_LENGTH, len(ALPHABET)))
for i in range(len(control_motif_dist)):
    control_motif_dist[i][i] +=1.0
lengths = []
for key in apd_data.keys():
    for j in range(len(apd_data[key])):
        lengths.append(key)

control_peps = []
for i in range(15):
    length = 5#np.random.choice(lengths)
    pep = make_controlled_pep(control_bg_dist, control_motif_dist, CONTROL_MOTIF_POS, length)
    control_peps.append(pep)

with open('control_peptides.txt', 'w+') as f:
    for peptide in control_peps:
        f.writelines((peptide))

