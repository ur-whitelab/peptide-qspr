import sys

ALPHABET = ['A','R','N','D','C','Q','E','G','H','I',
            'L','K','M','F','P','S','T','W','Y','V']

def read_data(datafile):
    '''Takes a file of peptide sequences and returns a list of counts corresponding
       to the observed amino acids in the file. I.e., generates the distro from the data.'''
    counts = [0 for item in ALPHABET]
    lengths = []
    with open(datafile, 'r') as f:
        lines = set(f.readlines())
    for line in lines:
        count_AAs(line, counts, lengths)
    return(counts, lengths)

def count_AAs(peptide, counts_list, length_list):
    '''Takes in a peptide (as a string) and a list that holds counts of AA occurrences.
       Fills the list appropriately.'''
    for item in peptide:
        if item in ALPHABET:
            counts_list[ALPHABET.index(item)] +=1
        length_list.append(len(peptide))
    return()

def printHelp():
    print("prepare_dists.py [peptide_list] [job] [aa_dist outfile] [length_dist outfile]")

def main():
    if(len(sys.argv) != 5):
        printHelp()
        exit()
    infile = sys.argv[1]
    job = sys.argv[2]
    aa_outfile = sys.argv[3]
    length_outfile = sys.argv[4]

    counts, lengths = read_data(infile)
    tot_count = sum(counts)
    aa_distro = [float(item)/float(tot_count) for item in counts]
    with open(aa_outfile, 'w+') as f:
        for item in aa_distro:
            f.write(str(item) +'\n' )
    with open(length_outfile, 'w+') as f:
        for item in lengths:
            f.write(str(item) + '\n')
    
main()
