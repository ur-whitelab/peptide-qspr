'''This program fits gaussians to the descriptor data provided. Must include both
   a "true" set, i.e. the APD or human dataset, and the fake data generated as you choose.'''
import pymc3 as pm
from copy import copy
import theano.tensor as tt
import numpy as np
import math
import sys
from qspr_plots.qspr_plots import *

def printHelp():
    print("APD_gaussmix.py [output_dirname] [APD_file_path] [fake_data_file_path] [NUM_CLUSTERS]")


    
def main():
    if len(sys.argv) != 5:
        printHelp()
        exit()
    
    homedir = sys.argv[1]
    infile = sys.argv[2]
    fakefile = sys.argv[3]
    NUM_CLUSTERS = sys.argv[4]
    
    data = []
    with open(infile, 'r') as datafile:
        data=datafile.readlines()
        
    #print((data[1].replace('\n','').split(', ')))
    
    for i in range(len(data)):
        data[i] = data[i].replace('\n','').replace(' ','').replace('"','').split(',')
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
    NUM_CLUSTERS = 2#the number of peaks each characteristic will have
    NSAMPLES = 2000
    #mixture of gaussians, each is 1d for starters, each with own covariance.
    with model:
        p = [pm.Dirichlet('{}_p'.format(key), a=np.ones(NUM_CLUSTERS), shape = NUM_CLUSTERS) for key in keys[1:]]

        p_min_potential = [pm.Potential('{}_p_min_potential'.format(key), tt.switch(tt.min(p[keys.index(key)-1]) < .1, -np.inf, 0)) for key in keys[1:]]
        # measurement error
        sd_priors = [pm.Uniform('{}_sd'.format(key), lower=0, upper = 20) for key in keys[1:]]
        
        #normal means
        mu_priors = [pm.Normal('{}_mu'.format(key), [30 for i in range(NUM_CLUSTERS)],
                               sd = 15, shape = NUM_CLUSTERS) for key in keys[1:]]
    
        category = [pm.Categorical('{}_category'.format(key),
                                  p=p[keys.index(key)-1],
                                   shape=len(trainsets['train_nHBAcc']), testval=0) for key in keys[1:]]
        
        #the likelihoods
        points = [pm.Normal('{}_observed'.format(key), mu = mu_priors[keys.index(key) -1][category], 
                             sd = sd_priors[keys.index(key) -1], 
                             observed = trainsets['train_{}'.format(key)])
                  for key in keys[1:]
                 ]
    
    
    
        #step1 = pm.Metropolis(vars=[p, sd_priors, mu_priors])
        #step2 = pm.ElemwiseCategorical(vars=[category], values=[[0,1] for key in keys[1:]])
        trace = pm.sample(NSAMPLES)#, step = [step1, step2])
    
    
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
    for key in occurrence_histograms.keys():
        np.savetxt('{}/data/gaussmix/{}_clusters_{}.counts'.format( homedir, NUM_CLUSTERS, key), occurrence_histograms[key][0])
        np.savetxt('{}/data/gaussmix/{}_clusters_{}.bins'.format( homedir, NUM_CLUSTERS, key), occurrence_histograms[key][1])
    
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
        bins = occurrence_histograms['{}_observed'.format(key)][1]
        counts = occurrence_histograms['{}_observed'.format(key)][0]
        for i in range(len(fakeset_probs)):
            fakeset_probs[i] += get_hist_prob(bins, counts,
                                             float(fakesets['{}'.format(key)][i])) 
    fakeset_probs /= len(devsets)
    
    print(fakeset_probs)
    fakemin, fakemax = min(fakeset_probs), max(fakeset_probs)
    print('fake dataset min/max: ', min(fakeset_probs), max(fakeset_probs))
    
    
    
    roc_min, roc_max = min(devmin, trainmin, fakemin),  max(devmax, trainmax, fakemax)
    print('lowest min: ', roc_min, 'highest max: ', roc_max)
    
    FPR_ARR, TPR_ARR, _, CUTOFF, BEST_IDX = gen_roc_data(npoints = 5000, roc_min = roc_min, roc_max = roc_max, fakes = fakeset_probs, devs = devset_probs, trains= trainset_probs)
    print("best cutoff value: {}".format(CUTOFF))
    print("using {} as our cutoff, we achieved an FPR of {} and a TPR of {}".format(CUTOFF, FPR_ARR[BEST_IDX], TPR_ARR[BEST_IDX]))
    #print('false positive rates: ', FPR_ARR, '\n','true positive rates: ', TPR_ARR)
    
    np.savetxt('{}/{}_clusters_FPR.txt'.format(homedir, NUM_CLUSTERS), FPR_ARR)
    np.savetxt('{}/{}_clusters_TPR.txt'.format(homedir, NUM_CLUSTERS), TPR_ARR)


if __name__ == '__main__':
    main()
