class qspr_plots:
    def __init__(self):
        '''Constructor for plotting class, with some constants defined.'''
        self.ALPHABET = ['A','R','N','D','C','Q','E','G','H','I', 'L','K','M','F','P','S','T','W','Y','V']
        
    def pep_to_int_list(self, pep):
        '''Takes a single string of amino acids and translates to a list of ints'''
        return(list(map(ALPHABET.index, pep.replace('\n', ''))))


    def get_hist_prob(self, bins, counts, value):
        '''takes in the bins and counts from a histogram (not necessarily normalized)
           and a value, returns the height of the bin that value falls into.'''
        idx = np.argmax(bins > value)
        if idx==0:
            return(0)
        else:
            return(counts[idx-1]/np.sum(counts))
