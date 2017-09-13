import numpy as np
import pymc3 as pm
import re
import math
import matplotlib as mpl
from matplotlib import pyplot as plt

#CONSTANTS
MOTIF_LENGTH = 4 #fixed motif lengths, for now
ALPHABET = ['A','R','N','D','C','Q','E','G','H','I',
            'L','K','M','F','P','S','T','W','Y','V']
regex = '\[(.*?)\]'#for processing the string mess 
fakefile = 'data/pdb_distributed_apd_length_peps.txt'

def pep_to_int_list(pep):
    '''takes a single string of amino acids and translates to a list of ints'''
    return(list(map(ALPHABET.index, pep)))
'''
def read_data(datafile, motif_file=None):
    #Takes a properly-formatted peptide datafile (each line MUST start with a sequence)
       #and reads it into a list.
    train_data = {}#dict keyed by peptide length containing the sequences
    test_data = {}#reserved testing data
    with open(datafile, 'r') as f:
        big_aa_string = ''#for training the whole background distro
        lines = f.readlines()
        nlines = len(lines)
        for line in lines[1:int(9*nlines/10)]:#skip the header
            pep = line.split(',')[0]
            length = len(pep)
            big_aa_string+=pep
            if(length not in train_data.keys()):
                train_data[length] = [(pep_to_int_list(pep))]
            else:
                train_data[length].append((pep_to_int_list(pep)))
        for line in lines[int(9*nlines/10) :]:
            pep = line.split(',')[0]
            length = len(pep)
            big_aa_string+=pep
            if(length not in test_data.keys()):
                test_data[length] = [(pep_to_int_list(pep))]
            else:
                test_data[length].append((pep_to_int_list(pep)))
        big_aa_list = pep_to_int_list(big_aa_string)
    return(train_data, test_data, big_aa_list)
'''

def read_data(datafile, motif_file=None):
    '''Takes a properly-formatted peptide datafile (each line MUST start with a sequence)
       and reads it into a list.'''
    data = {}#dict keyed by peptide length containing the sequences
    with open(datafile, 'r') as f:
        big_aa_string = ''#for training the whole background distro
        lines = f.readlines()
        nlines = len(lines)
        for line in lines[1:]:#skip the header
            pep = line.split(',')[0]
            length = len(pep)
            big_aa_string+=pep
            if(length not in data.keys()):
                data[length] = [(pep_to_int_list(pep))]
            else:
                data[length].append((pep_to_int_list(pep)))
        big_aa_list = pep_to_int_list(big_aa_string)
    return(data, big_aa_list)


def getmatches(pep, regex = regex):
    vals = []
    for match in re.findall( regex, pep):
        if match is not '':
            for item in match.split(' '):
                vals.append(item)
    return(vals)


def get_tot_prob(peptide, bg_hist, motif_start_hist, motif_hist_dict):
    '''Takes in a single peptide as a LIST OF INTS, the background histogram, the motif start
       histogram, and the dict of motif histograms, and returns the total probability
       density assigned to the sequence by the trained model.'''
    length = len(peptide)
    probs = np.zeros(length)#to hold the total prob of that letter in that spot
    #loop over all possible motif positions, offset by 1 due to bins
    end = min(length, max(motif_hist_dict.keys())-MOTIF_LENGTH)
    for i in range(0, end):
        prob = 0.0
        prob_not_motif = 0.0 #total prob a given index isn't in a motif
        prob_in_motif = 0.0 #total prob a given index is in a motif
        #loop over all places motif could start while including this position
        motif_idx = MOTIF_LENGTH-1
        for j in range(i - MOTIF_LENGTH, i):#+1 due to range()'s behavior
            if(j < end - MOTIF_LENGTH):
                #the probability this AA is in a motif at all is the sum of probs
                #of a motif starting at all possible start positions that include it
                prob_in_motif += get_hist_prob(motif_start_hist,j+1)
                prob += get_hist_prob(motif_start_hist, j+1) * motif_hist_dict[length][motif_idx][peptide[i]]
            motif_idx-=1
        prob_not_motif = 1 - prob_in_motif
        prob += get_hist_prob(bg_hist, peptide[i]) * prob_not_motif
        probs[i] = prob
    #finish the rest of the letters, drawing from the background dist only
    for i in range(end, length):#there's no chance they're in a motif.
        probs[i] = bg_hist[peptide[i]]
    return(np.sum(probs)/float(length))

def get_hist_prob(histogram, value):
    '''takes in a normalized histogram and a value. Returns the height of the bin that value falls into, or 0 if it is not within a bin.'''
    idx = np.argmax(histogram[1] > value)
    if idx==0:
        return(0)
    else:
        return(histogram[0][idx-1])

def calc_positives(arr, cutoff):
    '''takes in an array of probs given by the model and returns the number of
       probs above the given cutoff probability. For use in generating the ROC curve.'''
    arr = np.sort(np.array(arr))
    if arr[-1] > cutoff:
        return(len(arr) - np.argmax(arr >= cutoff))
    else:
        return(0)
    
def gen_roc_data(roc_min, roc_max, npoints, fakes,
                 trains):
    '''This fills two numpy arrays for use in plotting the ROC curve. The first is the FPR,
       the second is the TPR. The number of points used is npoints. 
       Returns (FPR_arr, TPR_arr, best_cutoff_value, and index_of_best_cutoff).'''
    best_cutoff = 0.0
    best_ROC = 0.0
    roc_range = np.linspace(roc_min, roc_max, npoints)
    fpr_arr = np.zeros(npoints)
    tpr_arr = np.zeros(npoints)
    #for each cutoff value, calculate the FPR and TPR
    for i in range(npoints):
        fakeset_positives = calc_positives(fakes, roc_range[i])
        fpr_arr[i] = float(fakeset_positives) / len(fakes)
        trainset_positives = calc_positives(trains, roc_range[i])
        tpr_arr[i] = float(trainset_positives) / (len(trains) )
    best_idx = 0
    old_dist = 2.0
    for i in range(0,npoints-1):
        dist = math.sqrt(fpr_arr[i] **2 + (1-tpr_arr[i]) **2)
        if (old_dist > dist):
            best_idx = i
            old_dist = dist
    best_cutoff = roc_range[best_idx]
    print('best index was {}'.format(best_idx))
    return( (fpr_arr, tpr_arr, best_cutoff, best_idx))


       
apd_data, all_apd_aa  = read_data('data/APD_qsar_new.txt')

peps = [] #empty list to fill with peptides...    


'''Train the model...'''

aa_model = pm.Model()
start_counts = [0 for i in range(200)]

with aa_model:
    bg_p = pm.Dirichlet('bg_p', a=np.ones(len(ALPHABET)), shape=len(ALPHABET))

    motif_p = pm.Dirichlet('motif_p', a=np.ones(len(ALPHABET)), shape=len(ALPHABET))
    
    motif_start = pm.DiscreteUniform('motif_start', 0, max(apd_data.keys()))

#    start_trace = pm.sample(len(apd_data.keys()), step = step1)

    for i in apd_data.keys():
        #randomly choose a starting position -- is this trained?!?


        start_loc = motif_start.random()[0]%(i-MOTIF_LENGTH)
        print(start_loc)
        obs = [apd_data[i][j][(start_loc):] for j in range(len(apd_data[i]))]
        motif_dist = pm.Categorical('motif_dist_{}'.format(i), p = motif_p,
                                    shape = MOTIF_LENGTH,
                                    observed = obs
                                   )
        '''motif_dist = pm.Categorical('motif_dist_{}'.format(i), p = motif_p,
                                    shape = MOTIF_LENGTH,
                                    observed = [
                                        apd_data[i][j][
                                        (start_trace['motif_start'][counter])%i
                                                      ]
                                        for j in range(len(apd_data[i]))
                                    ])'''
        bg_dist = pm.Categorical('bg_dist_{}'.format(i), p = bg_p,
                                 shape=(i), observed=all_apd_aa)
        #using the bg distro, get a good sampling
        step1 = pm.Metropolis(vars=[motif_start])        
        step2 = pm.CategoricalGibbsMetropolis(vars=[motif_dist, bg_dist])
        single_pep_trace = pm.sample(5000, step=[step1, step2])#, step=pm.Metropolis())#[step1, step2])
        #sample one whole peptide, once for each count of the given length.
        single_pep_ppc = pm.sample_ppc(single_pep_trace, samples=2000)#samples=len(apd_data[i]))
        for j in range(len(single_pep_ppc['bg_dist_{}'.format(i)])):
            peps.append(str(single_pep_ppc['bg_dist_{}'.format(i)][j][:start_loc]) +
                        str(single_pep_ppc['motif_dist_{}'.format(i)][j]) +
                        str(single_pep_ppc['bg_dist_{}'.format(i)][j][(start_loc + MOTIF_LENGTH):])
            )
        #peps.append(single_pep_ppc['motif_dist_{}'.format(i)])


'''Here onward we write the generated peptides to file...'''
peps = [pep.replace('\n', '') for pep in peps]#why does that show up??


intpeps = []#for storing the int lists
print('saving peptide file...')
with open('data/peptides/interactive_stepwise_motif_peps.txt', 'w+') as f:
    for pep in peps:
        arr = []
        values = getmatches(pep)
        for letter in values:
            if letter is not '' and letter is not '\n':
                arr.append(int(letter))
                f.write('{}'.format(ALPHABET[int(letter)]))
        intpeps.append(arr)
        f.write('\n')

apd_intpeps = []
for key in apd_data.keys():
    for item in apd_data[key]:
        apd_intpeps.append(item)


'''This fills a histogram of all the amino acids for each position of the motifs for each length of peptide we observe.'''
bg_histogram = (np.ones(20)) - 1.0#the background distribution, position-independent
motif_histograms = {}#the dict keyed by pep length for the motif associated with it
for key in apd_data.keys():
    motif_histograms[key] = []
    for i in range(MOTIF_LENGTH):
        motif_histograms[key].append(
            np.histogram(single_pep_ppc['motif_dist_{}'.format(key)][:,i:i+1],
                         bins=range(21), normed=True
            )[0] #bins are always the same, so only need to keep the first part
        )
    new_histogram = np.histogram(single_pep_ppc['bg_dist_{}'.format(key)],
                               bins = range(21), normed = True
                    )[0] #bins are always the same, so only need to keep the first part
    bg_histogram = bg_histogram + new_histogram

bg_histogram = bg_histogram/float(len(apd_data.keys()))
#the +2 is to deal with range() and np.histogram()'s behaviors
start_histogram = np.histogram(single_pep_trace['motif_start'], normed=True, bins=100)

#now to get the probabilities for each pep and construct an ROC curve

fake_data = []
with open(fakefile, 'r') as datafile:
    fake_data=datafile.readlines()
for i in range(len(fake_data)):
    fake_data[i] = fake_data[i].replace('\n','').replace(' ','').replace('"','')


fake_intpeps = [pep_to_int_list(item) for item in fake_data]
for i in range(10):
    for item in fake_intpeps:
        if(len(item)) not in apd_data.keys():
            fake_intpeps.remove(item)

fake_probs = [get_tot_prob(item, bg_histogram, start_histogram, motif_histograms) for item in fake_intpeps]
apd_probs = [ get_tot_prob(item, bg_histogram, start_histogram, motif_histograms) for item in intpeps]

rocmin = min(min(fake_probs), min(apd_probs))
rocmax = max(max(fake_probs), max(apd_probs))
npoints = 200
plotname = 'data/motif_ROC.png'

fpr_arr, tpr_arr, cutoff, best_idx = gen_roc_data(rocmin, rocmax, npoints, fake_probs, apd_probs)


plt.figure()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.plot(fpr_arr[:-1], tpr_arr[:-1], 'o', label='ROC at varied cutoffs', color='red')
plt.plot(fpr_arr[best_idx], tpr_arr[best_idx], 's', label='Best cutoff', color='blue')
plt.plot(fpr_arr, fpr_arr, label='totally random', ls=':', color='black')
plt.legend(loc='best')
plt.savefig(plotname)
