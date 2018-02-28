'''This program fits gaussians to the descriptor data provided. Must include both
   a "true" set, i.e. the APD or human dataset, and the fake data generated as you choose.'''
import pymc3 as pm
from copy import copy
import theano.tensor as tt
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from qspr_plots import *


def printHelp():
    print("APD_gaussmix.py [APD_file] [job] [fake_data_file] [log_file] [ROC_plot_filename] [optional:motif_file_to_test]")

    
if(len(sys.argv) != 6 and len(sys.argv) != 7):
    printHelp()
    exit()


infile = sys.argv[1]
job = sys.argv[2]
fakefile = sys.argv[3]
logfile = sys.argv[4]
plotname = sys.argv[5]
if(len(sys.argv) == 7):
    motiffile=sys.argv[6]
else:
    motiffile=fakefile

data = []
with open(infile, 'r') as datafile:
    data=datafile.readlines()
    
#print((data[1].replace('\n','').split(', ')))

for i in range(len(data)):
    data[i] = data[i].replace('\n','').replace(' ','').split(',')
print('the categories are: ', data[0])
print('example: ', data[1])

train_sequences = [item[0] for item in data[1:int(9/10 * len(data))]]
dev_sequences = [item[0] for item in data[int(9/10 * len(data)):]]
train_data = data[1:int(9/10 * len(data))]
dev_data = data[(int(9/10*len(data))):]

keys = copy(data[0])
keys.remove('MW')
devkeys=["dev_{}".format(key) for key in keys[1:]]
trainkeys=["train_{}".format(key) for key in keys[1:]]

devsets = {}
trainsets = {}

for key in trainkeys:
    for string in data[0]:
        if(key.find(string)):#we need the category
            idx = data[0].index(string)#this will be the column index
            trainsets[key] = [train_data[i][idx] for i in range(len(train_data))]
            
for key in trainkeys:
    assert(len(trainsets[key]) == len(train_sequences))
    
for key in devkeys:
    for string in data[0]:
        if(key.find(string)):
            idx = data[0].index(string)
            devsets[key] = [dev_data[i][idx] for i in range(len(dev_data))]

for key in devkeys:
    assert(len(devsets[key]) == len(dev_sequences))

    
model = pm.Model()
NUM_CHARACTERISTICS = 12#between 1 and 12; don't use MW
#mixture of gaussians, each is 1d for starters, each with own covariance.
with model:
    #no clustering, just positive cases
    # measurement error
    sd_priors = [pm.Uniform('{}_sd'.format(key), lower=0, upper = 20) for key in keys[1:]]
    
    #normal means
    mu_priors = [pm.Normal('{}_mu'.format(key), 0, sd = 15) for key in keys[1:]]
    
    #the likelihoods
    points = [pm.Normal('{}_observed'.format(key), mu = mu_priors[keys.index(key) -1], 
                         sd = sd_priors[keys.index(key) -1], 
                         observed = trainsets['train_{}'.format(key)])
              for key in keys[1:]
             ]

NSAMPLES = 5000
with model:
    trace = pm.sample(NSAMPLES)


#Do PPC sample first, get probs from that...
histograms=[]
i = 0
for key in keys[1:]:
    histograms.append(np.histogram(trace['{}_mu'.format(key)], bins=100))

with model:
    ppc = pm.sample_ppc(trace = trace, samples = NSAMPLES)

#fill a dict with the ppc histograms for easier use
occurrence_histograms = {}
for item in ppc.items():
    occurrence_histograms[item[0]] = np.histogram(item[1], bins='auto')
#print(occurrence_histograms['ALogP_observed'])
#print(np.sum(occurrence_histograms['ALogP_observed'][0]))

#write the histograms to file
with open(logfile, 'w+') as f:
    f.write('#THE DESCRIPTOR HISTOGRAMS FROM OUR PPC\n\n')
    for key in occurrence_histograms.keys():
        f.write('#{}\n'.format(str(key)))
        for item in occurrence_histograms[key]:
            f.write(str(item)[1:-1] + '\n')#skip the brackets...

#Now that we can get the histogram height given a single value as input, 
# we just need to loop through our devset and for each single peptide, get the total probability,
# which is just 1/N_DESCRIPTORS * (sum_over_descriptor_probs)
# then we can generate an ROC curve by getting false negative to false positive rates w/ our fake data and devset
devset_probs = np.zeros(len(devsets['dev_nHBAcc']))
trainset_probs = np.zeros(len(trainsets['train_nHBAcc'])) #to see what the probs are like on these
for key in keys[1:]:
    bins = occurrence_histograms['{}_observed'.format(key)][1]
    counts = occurrence_histograms['{}_observed'.format(key)][0]
    for i in range(len(devset_probs)):
        devset_probs[i] += get_hist_prob(bins, counts,
                                         float(devsets['dev_{}'.format(key)][i]))
    bins = occurrence_histograms['{}_observed'.format(key)][1]
    counts = occurrence_histograms['{}_observed'.format(key)][0]
    for i in range(len(trainset_probs)):
        trainset_probs[i] += get_hist_prob(bins, counts,
                                         float(trainsets['train_{}'.format(key)][i]))
devset_probs /= len(devsets)
trainset_probs /= len(trainsets)

devmin, devmax = min(devset_probs), max(devset_probs)
print('devset min/max: ', min(devset_probs), max(devset_probs))

trainmin, trainmax = min(trainset_probs), max(trainset_probs)
print('trainset min/max: ', min(trainset_probs), max(trainset_probs))

fake_data = []
#this is the fake dataset made up of peptides set to be about the same average
# length as that of the APD set. They are random sequences weighted by the distribution
# of amino acids among all proteins in the PDB.
with open(fakefile, 'r') as datafile:
    fake_data=datafile.readlines()
    

for i in range(len(fake_data)):
    fake_data[i] = fake_data[i].replace('\n','').replace(' ','').replace('"','').split(',')
    
fakesets={}
    
for key in keys:
    for string in fake_data[0]:
        if(key.find(string)):#we need the category
            idx = fake_data[0].index(string)#this will be the column index
            fakesets[key] = [fake_data[i][idx] for i in range(1,len(fake_data))]
#print(fakesets)

print(len(fakesets['nHBAcc']))

fakeset_probs = np.zeros(len(fakesets['nHBAcc']))
for key in keys[1:]:
    bins =occurrence_histograms['{}_observed'.format(key)][1]
    counts = occurrence_histograms['{}_observed'.format(key)][0]
    for i in range(len(fakeset_probs)):
        fakeset_probs[i] += get_hist_prob(bins, counts,
                                         float(fakesets['{}'.format(key)][i])) 
fakeset_probs /= len(devsets)

print(fakeset_probs)
fakemin, fakemax = min(fakeset_probs), max(fakeset_probs)
print('fake dataset min/max: ', min(fakeset_probs), max(fakeset_probs))



#Now that we have the sets of 'probabilies of being antimicrobial' we can generate our ROC curve
#We know that all the peptides in the fake set should be marked negative
# and all the ones in the training and dev sets should be marked positive
#From this, we can calculate the false positive and true positive rates for a variety of cutoffs.

roc_min, roc_max = min(devmin, trainmin, fakemin),  max(devmax, trainmax, fakemax)
print('lowest min: ', roc_min, 'highest max: ', roc_max)



FPR_ARR, TPR_ARR, _, CUTOFF, BEST_IDX = gen_roc_data(npoints = 5000, roc_min=roc_min, roc_max=roc_max, fakes = fakeset_probs, devs = devset_probs, trains = trainset_probs)
print("best cutoff value: {}".format(CUTOFF))
print("using {} as our cutoff, we achieved an FPR of {} and a TPR of {}".format(CUTOFF, FPR_ARR[BEST_IDX], TPR_ARR[BEST_IDX]))
#print('false positive rates: ', FPR_ARR, '\n','true positive rates: ', TPR_ARR)
        
plt.figure()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.plot(FPR_ARR[:-1], TPR_ARR[:-1], 'o', label='ROC at varied cutoffs', color='red')
plt.plot(FPR_ARR[BEST_IDX], TPR_ARR[BEST_IDX], 's', label='Best cutoff = {:.4}'.format(CUTOFF), color='blue')
plt.plot(FPR_ARR, FPR_ARR, label='totally random', ls=':', color='black')
plt.legend(loc='best')

plt.savefig(plotname)



#now for the motif tests
motif_data = []
#this is the dataset made up of peptides generated by the 'motif' model i made
with open(motiffile, 'r') as datafile:
    motif_data=datafile.readlines()
    

for i in range(len(motif_data)):
    motif_data[i] = motif_data[i].replace('\n','').replace(' ','').replace('"','').split(',')
    
motifsets={}
    
for key in keys:
    for string in motif_data[0]:
        if(key.find(string)):#we need the category
            idx = motif_data[0].index(string)#this will be the column index
            motifsets[key] = [motif_data[i][idx] for i in range(1,len(motif_data))]
#print(fakesets)

print(len(motifsets['nHBAcc']))

motifset_probs = np.zeros(len(motifsets['nHBAcc']))
for key in keys[1:]:
    bins = occurrence_histograms['{}_observed'.format(key)][1]
    counts = occurrence_histograms['{}_observed'.format(key)][0]
    for i in range(len(motifset_probs)):
        motifset_probs[i] += get_hist_prob(bins, counts,
                                         float(motifsets['{}'.format(key)][i])) 
motifset_probs /= len(devsets)

print('number of motif-generated peps predicted by this model to be AMP: {}'.format(
    np.array((motifset_probs) > CUTOFF, dtype=bool).sum()))
motifmin, motifmax = min(motifset_probs), max(motifset_probs)
print('motif dataset min/max: ', min(motifset_probs), max(motifset_probs))
