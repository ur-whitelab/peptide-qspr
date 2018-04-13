from numpy import genfromtxt, ones
from sys import argv
from math import sqrt
from scipy.stats import norm
from qspr_plots import *


def printhelp():
    print("Usage: evaluate_peptide.py [data_dir] [gaussmix_directory] [num_gauss_clusters] [gibbs_directory] [num_motif_classes] [motif_length] [peptide]")
    exit(1)

if(len(argv) != 8):
    printhelp()
    
DATA_DIR = argv[1]
quantile_means_file = DATA_DIR + "/baseline_means.csv"
quantile_vars_file = DATA_DIR + "/baseline_vars.csv"
GAUSSDIR = argv[2]
NUM_CLUSTERS = int(argv[3])
GIBBSDIR = argv[4]
NUM_MOTIF_CLASSES = int(argv[5])
MOTIF_LENGTH = int(argv[6])
PEPTIDE = argv[7]

NUM_DESCRIPTORS_ORIGINALLY_USED = 11 #don't change this!
N = NUM_DESCRIPTORS_ORIGINALLY_USED

keys = ['netCharge', 'nChargedGroups', 'nNonPolarGroups']#the 3 key descriptors

#load means and variances used for initial quantiling
quantile_means = {}
quantile_vars ={}
with open(quantile_means_file) as f:
    lines = f.readlines()
for line in lines:
    key = line.split(',')[0]
    val = float(line.split(',')[1].replace('\n',''))
    quantile_means[key] = val
with open(quantile_vars_file) as f:
    lines = f.readlines()
for line in lines:
    key = line.split(',')[0]
    val = float(line.split(',')[1].replace('\n',''))
    quantile_vars[key] = val

#load matrix of descriptor vals for each AA
with open(DATA_DIR + '/relevant_base_matrix.csv', 'r') as f:
    lines = f.readlines()

aa_values = {}
for line in lines[1:]:
    key = line.split(',')[0]
    vals = line.split(',')[1:]
    vals[-1] = vals[-1].replace('\n','')
    aa_values[key] = {}
    for i, item in enumerate(keys):
        aa_values[key][item] = int(vals[i])

def to_quantile(x, n, mean, var):
    val = norm.pdf(x, loc=mean*n, scale = sqrt(n) * sqrt(var) )
    return(val * 100.0)

with open(DATA_DIR + '/{}_clusters_{}_motifs_length_{}_combined_statistics_log.txt'.format(NUM_CLUSTERS, NUM_MOTIF_CLASSES, MOTIF_LENGTH), 'r') as f:
    lines = f.readlines()

opt_acc = 100. * float(lines[2].split()[2].replace('\n', '').replace('%', ''))
opt_motif_weight = float(lines[3].split()[3].replace('\n', '').replace('%', ''))
opt_qspr_weight =  float(lines[4].split()[3].replace('\n', '').replace('%', ''))
biggest_gibbs = float(lines[5].split()[3].replace('\n',''))
biggest_gauss = float(lines[6].split()[3].replace('\n',''))
lowest_gibbs =  float(lines[7].split()[3].replace('\n',''))
lowest_gauss =  float(lines[8].split()[3].replace('\n',''))
opt_cutoff = float(lines[9].split()[3].replace('\n',''))

print('optimal motif weight: {:.3}, optimal qspr weight: {:.3}, biggest gibbs: {:.3}, biggest gauss: {:.3}'.format(opt_motif_weight, opt_qspr_weight, biggest_gibbs, biggest_gauss))


bg_dist = genfromtxt('{}/bg_dist.txt'.format(GIBBSDIR))
motif_dists = ones((NUM_MOTIF_CLASSES, MOTIF_LENGTH, len(ALPHABET))) / float(len(ALPHABET))
for i in range(NUM_MOTIF_CLASSES):
    for j in range(MOTIF_LENGTH):
        motif_dists[i][j] = genfromtxt('{}/class_{}_of_{}_position_{}_motif_dist.txt'.format(GIBBSDIR,i,NUM_MOTIF_CLASSES, j))

with open(GIBBSDIR + '/motif_lists.txt', 'r') as f:
    lines = f.readlines()
motifs_list = lines[1::2]
motifs_list = [item.replace('\n', '') for item in motifs_list]
print('motifs list is {}'.format(motifs_list))


counts = {}
bins = {}
for key in keys:
    counts[key] = genfromtxt('{}/{}_clusters_{}_observed.counts'.format(GAUSSDIR, NUM_CLUSTERS, key))
    bins[key] = genfromtxt('{}/{}_clusters_{}_observed.bins'.format(GAUSSDIR, NUM_CLUSTERS, key))

#get this peptide's values for our three keys
pep_quant_scores = {}
for key in keys:
    pep_quant_val = 0.0
    for AA in PEPTIDE:
        pep_quant_val += aa_values[AA][key]#sum up the values
    pep_quant_scores[key] = to_quantile(pep_quant_val, len(PEPTIDE), quantile_means[key], quantile_vars[key])
    
prob = 0.0
for key in keys:
    prob += get_hist_prob(bins[key], counts[key], pep_quant_scores[key])/len(keys)
pep_gauss_prob = prob

pep_gibbs_prob = calc_prob(pep_to_int_list(PEPTIDE), bg_dist, motif_dists, num_motif_classes=NUM_MOTIF_CLASSES, motif_length=MOTIF_LENGTH)

print('pep_gibbs_prob is {} and pep_gauss_prob is {}'.format(pep_gibbs_prob, pep_gauss_prob))
scaled_gibbs_prob = pep_gibbs_prob / biggest_gibbs
scaled_gauss_prob = pep_gauss_prob / biggest_gauss

start_probs = []
for i in range(len(PEPTIDE) - MOTIF_LENGTH + 1):
    start_probs.append(calc_prob(pep_to_int_list(PEPTIDE), bg_dist, motif_dists, motif_start=i, num_motif_classes=NUM_MOTIF_CLASSES, motif_length=MOTIF_LENGTH))

motif_class_probs = []
for i in range(NUM_MOTIF_CLASSES):
    motif_class_probs.append(calc_prob(pep_to_int_list(PEPTIDE), bg_dist, motif_dists, motif_start=None, motif_class=i, num_motif_classes=NUM_MOTIF_CLASSES, motif_length=MOTIF_LENGTH))

most_likely_start = start_probs.index(max(start_probs))
most_likely_motif_idx = motif_class_probs.index(max(motif_class_probs))
most_likely_motif = motifs_list[most_likely_motif_idx]
found_motif = PEPTIDE[most_likely_start:most_likely_start+MOTIF_LENGTH]

print('This model predicts that the motif class most likely shown in this peptide was {}'.format(most_likely_motif))
print('This model predicts the most likely motif position in this peptide was {} (motif: {})'.format(most_likely_motif_idx, found_motif))

weighted_tot_prob = opt_qspr_weight * max(pep_gauss_prob - lowest_gauss, 0.0) / biggest_gauss + opt_motif_weight * max(pep_gibbs_prob - lowest_gibbs, 0.0) / biggest_gibbs

print('combined weighted prob is {:.4}'.format(weighted_tot_prob))

if weighted_tot_prob >= opt_cutoff:
    antimicrobial = True
else:
    antimicrobial = False

if antimicrobial:
    print('This model ({}% accuracy) predicts that this peptide could be antimicrobial!'.format(opt_acc))
else:
    print('This model ({}% accuracy) predicts that this peptide is probably not antimicrobial.'.format(opt_acc))
