import numpy as np
import re
import math
import matplotlib as mpl
from matplotlib import pyplot as plt
import pdb
import sys
from qspr_plots import *

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

DATA_DIR = '/home/rainier/pymc3_qspr/data/'
fakefile = DATA_DIR + 'pdb_distributed_apd_length_peps.txt'

def read_data(datafile, motif_file=None):
    '''Takes a properly-formatted peptide datafile (each line MUST start with a sequence)
       and reads it into a list.'''
    data = {}#dict keyed by peptide length containing the sequences
    with open(datafile, 'r') as f:
        big_aa_string = ''#for training the whole background distro
        lines = f.readlines()
        nlines = len(lines)
        start_idx = (1 if '#' in lines[0] else 0)
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
                for j in range(length - MOTIF_LENGTH):
                    for k in range(NUM_MOTIF_CLASSES):
                #we know where the motif is
                        if(i < motif_start or i >= motif_start + MOTIF_LENGTH):#\geq because of indexing
                            prob += bg_dist[peptide[i]] * start_dist[j] * class_dist[motif_class]
                        else:
                            prob += motif_dists[motif_class][ i - motif_start][peptide[i]] * start_dist[j] * class_dist[motif_class]
        else:#motif_class is None -> use distros
            for i in range(length):
                for j in range(length - MOTIF_LENGTH):
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
motif_class_counts = {}
for key in apd_data.keys():
    motif_start_dists[key] = np.ones((len(apd_data[key]), (key - MOTIF_LENGTH+1)))/float(key - MOTIF_LENGTH+1)
    motif_start_counts[key] = np.zeros((len(apd_data[key]), (key - MOTIF_LENGTH +1)), dtype = int)
    motif_class_dists[key] = np.ones((len(apd_data[key]) , NUM_MOTIF_CLASSES)) / float(NUM_MOTIF_CLASSES)
    motif_class_counts[key] = np.zeros((len(apd_data[key]) ,NUM_MOTIF_CLASSES))
bg_counts = np.zeros(len(ALPHABET), dtype=int)#times we see each AA as a b/g element
tot_bg_counts = np.zeros(len(ALPHABET), dtype=int)#times we see each AA as a b/g element

def do_bg_counts(peptide, bg_counts_arr, start):
    '''Takes in the bg_obs_counts array, a peptide, and the motif start positions,
       tallies up the number of times we see each AA as a background peptide. '''
    if(start == 0):
        for item in peptide[MOTIF_LENGTH:]:
            bg_counts_arr[item] += 1
    else:
        for item in peptide[0:start]:
            bg_counts_arr[item] += 1
        for item in peptide[(start+MOTIF_LENGTH):]:
            bg_counts_arr[item] += 1





#loop through each peptide, updating the distros NRUNS times

for _ in range(NRUNS):
    if(_%100 == 0):
        print('starting iteration {}'.format(_))
    for key in apd_data.keys():#loop over lengths of input peptides
        motif_counts[key] -= motif_counts[key]
        for i in range(len(apd_data[key])):#loop over all peptides of each length
            #bg_counts -= bg_counts #zero it out
            pep = apd_data[key][i]#the actual sequence
            possible_starts = np.array(range(key - MOTIF_LENGTH +1))#+1 b/c of range()
            motif_start = np.random.choice(possible_starts, p=motif_start_dists[key][i])
            motif_start_counts[key][i][motif_start]+=1
            #given our start position, calculate (count) the B/G and motif distros we observe
            do_bg_counts(pep, bg_counts, motif_start)
            #the raw counts are added directly, giving more weight to longer peptides
            #print('bg_counts is now {} with a sum of {}'.format(bg_counts, np.sum(bg_counts)))
            tot_bg_counts += bg_counts
            bg_dist += (bg_counts)
            motif_class = np.random.choice(range(NUM_MOTIF_CLASSES), p=motif_class_dists[key][i])
            for j in range(MOTIF_LENGTH):
                motif_counts[key][motif_class][j][pep[j+motif_start]] += 1
                tot_motif_counts[key][motif_class][j][pep[j+motif_start]] += 1
            for j in range(MOTIF_LENGTH):
                motif_dists[motif_class][j] += motif_counts[key][motif_class][j] 
    #normalize the distros at the END
    for key in apd_data.keys():
        for j in range(NUM_MOTIF_CLASSES):
            for k in range(MOTIF_LENGTH):
                #motif_dists[j][k] += float(len(ALPHABET))
                motif_dists[j][k] /= np.sum(motif_dists[j][k])
    #bg_dist /= np.sum(bg_dist)
    #now update the start probs based on observations...
    for key in apd_data.keys():#loop over all peps
        #motif_start_dists[key] -= motif_start_dists[key]
        for i in range(len(motif_start_dists[key])):#loop over all AA in pep
            for j in range(len(motif_start_dists[key][i])):#loop over all possible start positions
                #motif_start_dists[key][i][j] /= 2.0
                motif_start_dists[key][i][j] += get_tot_prob(
                    apd_data[key][i], bg_dist/np.sum(bg_dist),  motif_dists, motif_class_dists[key][i],
                    motif_start_dists[key][i], motif_start = j, motif_class=None
                )# * ETA#/2.0#would do motif_start_dists[key][i]
        for i in range(len(motif_start_dists[key])):
            #motif_start_dists[key][i] += float(len(motif_start_dists[key][i])) #add some 'noise'
            motif_start_dists[key][i] /= np.sum(motif_start_dists[key][i])
    #now update the motif class probs based on observations
    for key in apd_data.keys():
        #motif_class_dists[key] -= motif_class_dists[key]#zero out
        for i in range(len(motif_class_dists[key])):
            for j in range(len(motif_class_dists[key][i])):
                for k in range(key-MOTIF_LENGTH+1):
                #motif_class_dists[key][i][j] /= 2.0
                    motif_class_dists[key][i][j] += get_tot_prob(
                        apd_data[key][i], bg_dist/np.sum(bg_dist),  motif_dists,
                        class_dist=(motif_class_dists[key][i]), motif_class=j,
                        motif_start = k, start_dist=(motif_start_dists[key][i])
                )#* ETA#/2.0
        for i in range(len(motif_class_dists[key])):
            #motif_class_dists[key][i] += float(len(motif_class_dists[key][i]))
            motif_class_dists[key][i] /= np.sum(motif_class_dists[key][i])
    #pdb.set_trace()


#now get some fake data for performance analysis
'''fake_data = []
with open(fakefile, 'r') as datafile:
    fake_data=datafile.readlines()
for i in range(len(fake_data)):
    fake_data[i] = fake_data[i].replace('\n','').replace(' ','').replace('"','')

fake_intpeps = [pep_to_int_list(item) for item in fake_data]
#ugly workaround of python parsing...
for i in range(10):
    for item in fake_intpeps:
        if(len(item)) not in apd_data.keys():
            fake_intpeps.remove(item)
#process APD data into list also
apd_intpeps = []
for key in apd_data.keys():
    for item in apd_data[key]:
        apd_intpeps.append(item)

fake_probs = []
apd_probs = []
'''
#collapse the distros for starting and classes  for each length into one distro per length
collapsed_start_dists = {}
collapsed_class_dists = {}
for key in apd_data.keys():
    collapsed_start_dists[key] = np.sum(motif_start_dists[key], axis=0) / np.sum(motif_start_dists[key])
    collapsed_class_dists[key] = np.sum(motif_class_dists[key], axis=0) / np.sum(motif_class_dists[key])

'''for fakepep in fake_intpeps:
    fake_probs.append(get_tot_prob(
        fakepep, bg_dist,  motif_dists, class_dist=collapsed_class_dists[len(fakepep)],
        motif_class=None, motif_start = None, start_dist=collapsed_start_dists[len(fakepep)]
    ))

for pep in apd_intpeps:
    apd_probs.append(get_tot_prob(
        pep, bg_dist,  motif_dists, class_dist=collapsed_class_dists[len(pep)],
        motif_class=None, motif_start = None, start_dist=collapsed_start_dists[len(pep)]
    ))'''
    


'''rocmin = min( min(fake_probs), min(apd_probs) )
rocmax = max( max(fake_probs), max(apd_probs) )

npoints = 2000
    
fpr_arr, tpr_arr, cutoff, best_idx = gen_roc_data(rocmin, rocmax, npoints, fake_probs, apd_probs)

plotname = "gibbs_sampling_motif_length_{}_{}_classes_{}_iterations.png".format(MOTIF_LENGTH, NUM_MOTIF_CLASSES, NRUNS)
fig = plt.figure()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.plot(fpr_arr[:-1], tpr_arr[:-1], 'o', label='ROC at varied cutoffs', color='red')
plt.plot(fpr_arr[best_idx], tpr_arr[best_idx], 's', label='Best cutoff', color='blue')
plt.plot(fpr_arr, fpr_arr, label='totally random', ls=':', color='black')
plt.legend(loc='best')
plt.savefig(plotname)
plt.close(fig)'''

#make and save bg distros for sanity checks
fig = plt.figure()
plt.xlabel('Amino Acid')
plt.ylabel('Relative Frequency')
plt.title('Background Histogram After Sampling')
plt.bar(range(20), bg_dist/np.sum(bg_dist))
plt.xticks(range(20), ALPHABET)
plt.savefig('background_dist_{}_motif_classes_with_sampling.png'.format(NUM_MOTIF_CLASSES))
plt.close(fig)

all_peps_hist = np.histogram(all_apd_aa, bins=range(21), normed=True)
fig = plt.figure()
plt.xlabel('Amino Acid')
plt.ylabel('Relative Frequency')
plt.title('Background Histogram Without Sampling')
plt.bar(range(20), all_peps_hist[0])
plt.xticks(range(20), ALPHABET)
plt.savefig('background_dist_{}_motif_classes_no_sampling.png'.format(NUM_MOTIF_CLASSES))
plt.close(fig)

for i in range(NUM_MOTIF_CLASSES):
    for j in range(MOTIF_LENGTH):
        fig = plt.figure()
        plt.ylim(0,1.0)
        plt.title('position {} in motif class {}'.format(j, i))
        plt.xlabel('amino acid')
        plt.ylabel('relative occurrence rate')
        plt.bar(range(20), motif_dists[i][j])
        plt.xticks(range(20), ALPHABET)
        plt.savefig('class_{}_of_{}_position_{}_motif_dist.png'.format(i,NUM_MOTIF_CLASSES,j))
        plt.close(fig)
