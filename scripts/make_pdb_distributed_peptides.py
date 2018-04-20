import numpy as np
import sys
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

'''This script takes in a distribution file specifying the length distribution of peptides
 in some dataset, and makes random peptides with lengths drawn uniformly from the lengths given.'''

ALPHABET = ['A','R','N','D','C','Q','E','G','H','I',
            'L','K','M','F','P','S','T','W','Y','V']

def read_dists(aa_dist_file, length_file):
    '''Reads a distro file such as one created by prepare_pdb.py and returns the distros.'''
    aa_dist = np.genfromtxt(aa_dist_file)
    with open(length_file, 'r') as f:
        lengths = [int(line) for line in f]
    return(aa_dist, lengths)
    
    
def make_pep(distribution, lengths):
    '''Using a distribution of counts and lengths, generates a weighted random peptide.'''
    peptide = ''
    length_idx = np.random.randint(0, high=len(lengths))
    length = lengths[length_idx]
    for i in range(length-1):#not sure how I got off-by-one but this fixed it.
        choice = np.random.choice(np.arange(len(ALPHABET)), p=distribution)
        peptide+=(ALPHABET[choice])
    return(peptide+'\n')
    

def make_peptides(outfile, distro, lengths, npeps):
    '''Using the distribution in distro, generates npeps random peptides, and prints
       to outfile.'''
    count_tot = sum(distro)
    distro = [float(a)/float(count_tot) for a in distro]
    with open(outfile, 'w+') as f:
        for i in range(npeps):
            peptide = make_pep(distro, lengths)
            f.write(str(peptide))
        
def printHelp():
    print("make_pdb_distributed_peptides.py [aa_dist_file] [lengths_file] [job] [peptide number] [outfile]")

def main():
    
    if(len(sys.argv) != 6):
        printHelp()
        exit()

    aa_infile = sys.argv[1]
    length_infile = sys.argv[2]
    job = sys.argv[3]
    number = int(sys.argv[4]) #number of peps to make
    outfile = sys.argv[5]

    bg_dist, lengths = read_dists(aa_infile, length_infile)

    make_peptides(outfile, bg_dist, lengths, number)

main()
