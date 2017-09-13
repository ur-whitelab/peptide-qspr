import numpy as np

MOTIF_LENGTH = 4 #fixed motif lengths, for now
ALPHABET = ['A','R','N','D','C','Q','E','G','H','I',
            'L','K','M','F','P','S','T','W','Y','V']

def read_data(datafile, motif_file=None):
    '''Takes a properly-formatted peptide datafile and reads it into a numpy array.
       Optionally also treats a motif file, which should be one number per line,
       indicating the motif class to which each peptide specified within the datafile
       belongs.'''
    raw_data = np.genfromtxt(datafile)
    lengths = [0 for i in range((int(raw_data[-1][0]+1)))]
    data = []
    with open(datafile) as f:
        for i in range(len(f.readlines())):
            data.append(raw_data[i][2])
            lengths[int(raw_data[i][0])] = int(raw_data[i][3] + 1)
#    data = np.array(data)
    return(data, lengths)

apd_data, lengths  = read_data('data/APD.data')

#this is where each individual sequence starts, as indicated by their lengths.
start_indices = [ sum(lengths[:i]) for i in range(len(lengths))]

print( start_indices[0:5])

#list version:
observed_motifs = []
for i in range(len(start_indices)):
    #pick a random spot to start within the current peptide
    if i < (len(start_indices) -1):
        idx = np.random.randint(start_indices[i], start_indices[i+1]-MOTIF_LENGTH+1)
    else:#the last one
        idx = np.random.randint(start_indices[i], start_indices[i]+lengths[i]-MOTIF_LENGTH+1)
    observed_motifs.append(apd_data[idx:idx+MOTIF_LENGTH])
       

#dict version:

'''observed_motifs = {}
for i in range(len(start_indices)):
    #pick a random spot to start within the current peptide
    if i < (len(start_indices) -1):
        idx = np.random.randint(start_indices[i], start_indices[i+1]-MOTIF_LENGTH+1)
    else:#the last one
        idx = np.random.randint(start_indices[i], start_indices[i]+lengths[i]-MOTIF_LENGTH+1)
    if lengths[i] in observed_motifs.keys():
        observed_motifs[lengths[i]].append(apd_data[idx:idx+MOTIF_LENGTH])
    else:
        observed_motifs[lengths[i]] = [(apd_data[idx:idx+MOTIF_LENGTH])]
    
'''
#a check for that it's random and also works on the small ones...
print('the short motif as a test:',apd_data[start_indices[1111]:start_indices[1111]+lengths[1111]])
print( '\nthe motif from that shorty:', observed_motifs[1111])
