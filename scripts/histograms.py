import numpy as np
import pymc3 as pm
import matplotlib as mpl
import matplotlib.pyplot as plt

ALPHABET = ['A','R','N','D','C','Q','E','G','H','I',
            'L','K','M','F','P','S','T','W','Y','V']

def read_data(datafile):
    '''Takes a properly-formatted peptide datafile and reads it into a numpy array.
       Also records the lengths of all the amino acids seen, for histogramming.'''
    raw_data = np.genfromtxt(datafile)
    lengths = np.zeros((int(raw_data[-1][0]+1)))
    data = []
    with open(datafile) as f:
        for i in range(len(f.readlines())):
            data.append(raw_data[i][2])
            lengths[int(raw_data[i][0])] = raw_data[i][3] + 1
    data = np.array(data)
    return(data, lengths)

apd_data, lengths  = read_data('data/APD.data')

plt.figure()
plt.hist(lengths, bins=100)
plt.title('Histogram of APD Lengths')
plt.xlabel('length')
plt.ylabel('count')
plt.savefig('APD_length_histogram.png', transparent=True)

model = pm.Model()

with model:
    length_mean = pm.HalfNormal('length_mean', sd=25)
    length_mean2 = pm.Normal('length_mean', mu=75, sd=25)
    length = pm.Poisson('length', mu=length_mean, observed=lengths)
    length2 = pm.Normal('length2', mu=length_mean2, observed=lengths)
    trace = pm.sample(10000)
    ppc = pm.sample_ppc(trace)

print(ppc)
plt.figure()
plt.hist(ppc['length'] + ppc['length2'], bins=100)
plt.title('sampled lengths from pymc3 fitting')
plt.xlabel('length')
plt.ylabel('count')
plt.savefig('APD_length_histogram_from_sampling.png', transparent=True)
