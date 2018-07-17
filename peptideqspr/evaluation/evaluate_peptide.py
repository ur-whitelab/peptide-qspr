from numpy import genfromtxt, ones
from sys import argv
from math import sqrt
from scipy.stats import norm
from peptideqspr.qspr_plots.qspr_plots import *
import pkg_resources

#example use as script:

#python3 -i evaluate_peptide.py /home/rainier/pymc3_qspr/data/ /home/rainier/pymc3_qspr/data/gaussmix/gpos/only_3_descriptors/ 3 /home/rainier/pymc3_qspr/data/gibbs/with_starts/8_classes/gpos_8_classes_length_3/ 8 3 KLLKLLKLLKKLLLKLKLK

def printhelp():
    print('Usage: evaluate_peptide.py [peptide] [human (optional, default 0)]')
    exit(1)

class Model:
    def to_quantile(self, x, n, mean, var):
        val = norm.pdf(x, loc=mean*n, scale = sqrt(n) * sqrt(var) )
        return(val * 100.0)
            
    def read_data(self, human=False):
        self.quantile_means = {}
        self.quantile_vars ={}
        with open(self.quantile_means_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            key = line.split(',')[0]
            val = float(line.split(',')[1].replace('\n',''))
            self.quantile_means[key] = val
        with open(self.quantile_vars_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            key = line.split(',')[0]
            val = float(line.split(',')[1].replace('\n',''))
            self.quantile_vars[key] = val
        #load matrix of descriptor vals for each AA
        with open(self.DATA_DIR + '/relevant_base_matrix.csv', 'r') as f:
            lines = f.readlines()
        self.aa_values = {}
        for line in lines[1:]:
            key = line.split(',')[0]
            vals = line.split(',')[1:]
            vals[-1] = vals[-1].replace('\n','')
            self.aa_values[key] = {}
            for i, item in enumerate(self.keys):
                self.aa_values[key][item] = int(vals[i])

        if human:
            pathval = '/human'
        else:
            pathval = ''
        
        #load means and variances used for initial quantiling
        with open(self.DATA_DIR + pathval + '/{}_clusters_{}_motifs_length_{}_combined_statistics_log.txt'.format(self.NUM_CLUSTERS, self.NUM_MOTIF_CLASSES, self.MOTIF_LENGTH), 'r') as f:
            lines = f.readlines()

        #do all the calculations for the antimicrobial models
        self.opt_acc = 100. * float(lines[2].split()[2].replace('\n', '').replace('%', ''))
        self.opt_motif_weight = float(lines[3].split()[3].replace('\n', '').replace('%', ''))
        self.opt_qspr_weight =  float(lines[4].split()[3].replace('\n', '').replace('%', ''))
        self.biggest_gibbs = float(lines[5].split()[3].replace('\n',''))
        self.biggest_gauss = float(lines[6].split()[3].replace('\n',''))
        self.lowest_gibbs =  float(lines[7].split()[3].replace('\n',''))
        self.lowest_gauss =  float(lines[8].split()[3].replace('\n',''))
        self.opt_cutoff = float(lines[9].split()[3].replace('\n',''))
        with open('{}/bg_dist.txt'.format(self.GIBBSDIR), 'r') as f:
            self.bg_dist = genfromtxt(f)
        self.motif_dists = ones((self.NUM_MOTIF_CLASSES, self.MOTIF_LENGTH, len(ALPHABET))) / float(len(ALPHABET))
        for i in range(self.NUM_MOTIF_CLASSES):
            for j in range(self.MOTIF_LENGTH):
                with open('{}/class_{}_of_{}_position_{}_motif_dist.txt'.format(
                        self.GIBBSDIR,i,self.NUM_MOTIF_CLASSES, j), 'r') as f:
                    self.motif_dists[i][j] = genfromtxt(f)
        if not human:
            with open(self.GIBBSDIR + '/motif_lists.txt', 'r') as f:
                lines = f.readlines()
            self.motifs_list = lines[1::2]
            self.motifs_list = [item.replace('\n', '') for item in self.motifs_list]


        self.counts = {}
        self.bins = {}
        for key in self.keys:
            with open('{}/{}_clusters_{}_observed.counts'.format(self.GAUSSDIR, self.NUM_CLUSTERS, key), 'r') as f:
                self.counts[key] = genfromtxt(f)
            with open('{}/{}_clusters_{}_observed.bins'.format(self.GAUSSDIR, self.NUM_CLUSTERS, key), 'r') as f:
                self.bins[key] = genfromtxt(f)
            
    def setup(self, human = False):
        if not human:#for apd (default), use params for apd-trained model
            self.GAUSSDIR = pkg_resources.resource_filename(__name__, 'resources/gauss')
            self.NUM_CLUSTERS = 3
            self.GIBBSDIR = pkg_resources.resource_filename(__name__, 'resources/gibbs')
            self.NUM_MOTIF_CLASSES = 8
            self.MOTIF_LENGTH = 3
        else:#for human, use params for human-trained model
            self.GAUSSDIR = pkg_resources.resource_filename(__name__, 'resources/human/gauss')
            self.NUM_CLUSTERS = 8
            self.GIBBSDIR = pkg_resources.resource_filename(__name__, 'resources/human/gibbs')
            self.NUM_MOTIF_CLASSES = 0
            self.MOTIF_LENGTH = 0
    
    def __init__(self, human = False):
        self.DATA_DIR = pkg_resources.resource_filename(__name__, 'resources/')
        self.quantile_means_file = pkg_resources.resource_filename(__name__, 'resources/baseline_means.csv')
        self.quantile_vars_file = pkg_resources.resource_filename(__name__, 'resources/baseline_vars.csv')
        self.keys = ['netCharge', 'nChargedGroups', 'nNonPolarGroups']#the 3 key descriptors
        self.setup(human = human)
        self.read_data(human = human)

    def evaluate_peptide(self, peptide, human):
        retval = {'accuracy': self.opt_acc}#initialize blank dict
        if not peptide.isalpha():
            raise ValueError(
                'Bad input. These letters are valid amino acids: {}'.format(str(ALPHABET)[1:-1]))
        if not peptide.isupper():
            peptide = peptide.upper()
        
        #get this peptide's values for our three keys
        pep_quant_scores = {}
        for key in self.keys:
            pep_quant_val = 0.0
            for AA in peptide:
                pep_quant_val += self.aa_values[AA][key]#sum up the values
            pep_quant_scores[key] = self.to_quantile(pep_quant_val, len(peptide), self.quantile_means[key], self.quantile_vars[key])
    
        prob = 0.0
        for key in self.keys:
            prob += get_hist_prob(self.bins[key], self.counts[key], pep_quant_scores[key])/len(self.keys)
        pep_gauss_prob = prob

        pep_gibbs_prob = calc_prob(pep_to_int_list(peptide), self.bg_dist, self.motif_dists, num_motif_classes=self.NUM_MOTIF_CLASSES, motif_length=self.MOTIF_LENGTH)
        scaled_gibbs_prob = pep_gibbs_prob / self.biggest_gibbs
        scaled_gauss_prob = pep_gauss_prob / self.biggest_gauss

        start_probs = []
        for i in range(len(peptide) - self.MOTIF_LENGTH + 1):
            start_probs.append(
                calc_prob(
                    pep_to_int_list(peptide), self.bg_dist, self.motif_dists,
                    motif_start=i, num_motif_classes=self.NUM_MOTIF_CLASSES,
                    motif_length=self.MOTIF_LENGTH))

        motif_class_probs = []
        for i in range(self.NUM_MOTIF_CLASSES):
            motif_class_probs.append(
                calc_prob(
                    pep_to_int_list(peptide), self.bg_dist, self.motif_dists,
                    motif_start=None, motif_class=i, num_motif_classes=self.NUM_MOTIF_CLASSES,
                    motif_length=self.MOTIF_LENGTH))
        if not human:#the antifouling set has no motifs
            most_likely_start = start_probs.index(max(start_probs))
            most_likely_motif_idx = motif_class_probs.index(max(motif_class_probs))
            most_likely_motif = self.motifs_list[most_likely_motif_idx]
            found_motif = peptide[most_likely_start:most_likely_start+self.MOTIF_LENGTH]
            retval['most_likely_motif'] = most_likely_motif
            retval['most_likely_motif_idx'] = most_likely_motif_idx
            retval['most_likely_motif_start'] = most_likely_start
            retval['found_motif'] = found_motif

        gibbs_contributes_more = True

        weighted_gibbs_prob = self.opt_motif_weight * max(pep_gibbs_prob - self.lowest_gibbs, 0.0) / self.biggest_gibbs
        weighted_gauss_prob = self.opt_qspr_weight * max(pep_gauss_prob - self.lowest_gauss, 0.0) / self.biggest_gauss

        if weighted_gauss_prob > weighted_gibbs_prob:
            gibbs_contributes_more = False

        weighted_tot_prob = weighted_gibbs_prob + weighted_gauss_prob

        if gibbs_contributes_more:
            retval['qspr_contributes_more'] = False
        else:
            retval['qspr_contributes_more'] = True

        if weighted_tot_prob >= self.opt_cutoff:
            positive = True
        else:
            positive = False

        if not human:
            if positive:
                retval['predict'] = True
            else:
                retval['predict'] = False
        else:
            if positive:
                retval['predict'] = True
            else:
                retval['predict'] = False
        retval['score'] = weighted_tot_prob / self.opt_cutoff
        return(retval)
    def predict(self, peptide):
        #first get antimicrobial prediction
        self.setup(human = False)
        self.read_data(human = False)
        antimicrobial_predict = self.evaluate_peptide(peptide, human = False)
        #now get antifouling prediction
        self.setup(human = True)
        self.read_data(human = True)
        antifouling_predict = self.evaluate_peptide(peptide, human = True)
        retval = {}
        retval['antimicrobial'] = antimicrobial_predict
        retval['antifouling'] = antifouling_predict
        return(retval)


            



def main():
    if len(argv) != 2 and len(argv) != 3:
        printhelp()
    PEPTIDE = argv[1]
    if len(argv) == 3:
        HUMAN = bool(int(argv[2]))
    else:
        HUMAN = False
    model = Model(human = True)
    #antifouling_predict = model.evaluate_peptide(PEPTIDE, human = True)
    #model = Model(human = False)
    #antimicrobial_predict = model.evaluate_peptide(PEPTIDE, human = False)
    predict = model.predict(PEPTIDE)
    print('Done evaluating.')
    print('Antifouling: {}'.format(predict['antifouling']))
    print('Antimicrobial: {}'.format(predict['antimicrobial']))

    
if __name__ == '__main__':
    main()

