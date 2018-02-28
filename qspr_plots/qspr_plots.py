import numpy
ALPHABET = ['A','R','N','D','C','Q','E','G','H','I', 'L','K','M','F','P','S','T','W','Y','V']
        
def pep_to_int_list( pep):
    '''Takes a single string of amino acids and translates to a list of ints'''
    return(list(map(ALPHABET.index, pep.replace('\n', ''))))


def get_hist_prob( bins, counts, value):
    '''takes in the bins and counts from a histogram (not necessarily normalized)
    and a value, returns the height of the bin that value falls into.'''
    idx = numpy.argmax(bins > value)
    if idx==0:
        return(0)
    else:
        return(counts[idx-1]/numpy.sum(counts))


def read_logs( trainfile, testfile, return_strings = False):
    '''Reads in the log files generated by runs of the other python files.'''
    train_data = {}#dict keyed by peptide length containing the sequences
    test_data = {}
    peptide_strings = {}
    train_peptides = []
    test_peptides = []
    big_aa_string = ''#for training the whole background distro
    with open(trainfile, 'r') as f:
        lines = f.readlines()
        nlines = len(lines)
        start_idx = (1 if ('#' in lines[0] or 'sequence' in lines[0]) else 0)
        for line in lines[start_idx:]:#skip the header
            pep = line.split(',')[0]
            train_peptides.append(pep)
            length = len(pep)
            big_aa_string+=pep
            if(length not in train_data.keys()):
                train_data[length] = [(pep_to_int_list(pep))]
            else:
                train_data[length].append((pep_to_int_list(pep)))
            if(length not in peptide_strings.keys()):
                peptide_strings[length] = [pep]
            else:
                peptide_strings[length].append(pep)
    with open(testfile, 'r') as f:
        lines = f.readlines()
        nlines = len(lines)
        start_idx = (1 if ('#' in lines[0] or 'sequence' in lines[0]) else 0)
        for line in lines[start_idx:]:#skip the header
            pep = line.split(',')[0]
            test_peptides.append(pep)
            length = len(pep)
            big_aa_string+=pep
            if(length not in test_data.keys()):
                test_data[length] = [(pep_to_int_list(pep))]
            else:
                test_data[length].append((pep_to_int_list(pep)))
            if(length not in peptide_strings.keys()):
                peptide_strings[length] = [pep]
            else:
                peptide_strings[length].append(pep)
    big_aa_list = pep_to_int_list(big_aa_string)
    if(return_strings):
        return(train_peptides, test_peptides, train_data, test_data, big_aa_list, peptide_strings)
    else:
        return(train_peptides, test_peptides, train_data, test_data, big_aa_list)

def calc_positives(arr, cutoff):
    '''takes in an array of probs given by the above model and returns the number of
       probs above the cutoff probability. This is for use in generating the ROC curve.'''
    arr = np.sort(np.array(arr))
    if not arr[-1] < cutoff:
        return(len(arr) - np.argmax(arr > cutoff))
    else:
        return(0)

def gen_roc_data(npoints, roc_min, roc_max, fakes,
                 devs, trains):
    '''This fills two numpy arrays for use in plotting the ROC curve. The first is the FPR,
       the second is the TPR. The number of points is npoints. Returns (FPR_arr, TPR_arr).'''
    best_cutoff = 0.0
    best_ROC = 0.0
    roc_range = np.linspace(roc_min, roc_max, npoints)
    fpr_arr = np.zeros(npoints)
    tpr_arr = np.zeros(npoints)
    #for each cutoff value, calculate the FPR and TPR
    for i in range(npoints):
        fakeset_positives = calc_positives(fakes, roc_range[i])
        fpr_arr[i] = float(fakeset_positives) / len(fakes)
        devset_positives =  calc_positives(devs, roc_range[i])
        trainset_positives = calc_positives(trains, roc_range[i])
        tpr_arr[i] = float(devset_positives + trainset_positives) / (len(devs) + len(trains) )
    best_idx = 0
    old_dist = 2.0
    for i in range(0,npoints-1):
        dist = math.sqrt(fpr_arr[i] **2 + 2* (1-tpr_arr[i]) **2)
        if (old_dist > dist):
            best_idx = i
            old_dist = dist
    best_cutoff = roc_range[best_idx]
    print('best index was {}'.format(best_idx))
    accuracy = (tpr_arr[best_idx] + (1.0-fpr_arr[best_idx]))/2.0
    return( (fpr_arr, tpr_arr, accuracy, best_cutoff, best_idx))
