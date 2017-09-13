import numpy as np
from random import randint
ETA = 1.0#the initial learning rate
b = 0.0#The initial b value. Might as well start at 0
C = 0.2#The C value. Small for small changes to b
CUTOFF_NUMBER = 1000
N_FEATURES = 12 # the number of peptide descriptors we're using. by default, all 12
DATA_DIR = 'data/' #path to data directory

def read_file(filename, test_arr, dev_arr):
    '''Reads a specified file of the correct input format into target_arr for use in training.
        We only have positive examples in our training set, the APD, so all y-values are 1.0'''
    f = open(filename, 'r')
    lines = f.readlines()
    i = 0
    for line in lines[1:]:#don't need the headings for this one
        data = line.replace('\n','').replace(' ','').replace('NA','0').split(',')
        if("APD_qsar_new.txt" in filename):
            data.remove(data[2])
        data = data[2:]#don't need to track sequences either
        #print(i, len(data), data)
        if(i < len(test_arr)):
            test_arr[i] = data
        else:
            dev_arr[i-len(test_arr)] = data
        i+=1
    #print(test_arr, dev_arr)
            

print("Setting up...")
N_POS = 1784 - 178 #taken from the APD dataset's size, with 10% of data withheld.
N_POS_WITHHELD = 178 #the withheld data
N_NEG = 9955 - 955 #this is the fake data length.
N_NEG_WITHHELD = 955 # this is the withheld fake data length
N_TOT = N_POS + N_NEG
#array sizes entered manually, taken from `wc -l <file>` on the respective files
training_pos_arr = np.zeros((N_POS , N_FEATURES))
training_neg_arr = np.zeros((N_NEG , N_FEATURES))
dev_pos_arr = np.zeros(( N_POS_WITHHELD, N_FEATURES))
dev_neg_arr = np.zeros(( N_NEG_WITHHELD, N_FEATURES))
weight_vec = np.zeros(N_FEATURES)#the w vector, full of random floats 0 to 1


print("Reading training data from file...")
read_file(DATA_DIR+"APD_qsar_new.txt", training_pos_arr, dev_pos_arr)
read_file(DATA_DIR+"big_testpeps.out", training_neg_arr, dev_neg_arr )

print("Starting Support Vector Machine...")
done = False
nsteps = 0
while(done == False):
    num_mistakes = 0
    for i in range(N_POS):
        ans = 1 - ( 1.0 * ( np.dot( np.transpose(weight_vec) , training_pos_arr[i] ) + b) )
        if(np.sign(ans) == 1):#mistake, this should be negative
            num_mistakes += 1
            weight_vec -= ETA * ( weight_vec / (N_TOT) - C * 1.0 * training_pos_arr[i] )
        else:
            weight_vec -= ETA * weight_vec / N_TOT
        ans = 1 - ( 1.0 * ( np.dot( np.transpose(weight_vec) , training_pos_arr[i] ) + b) )
        if(np.sign(ans) == 1):
            b += C * ETA * 1.0
    for i in range(N_NEG):#-1 here because these are from the randomly generated peptides
        ans = 1 - ( -1.0 * ( np.dot( np.transpose(weight_vec) , training_neg_arr[i] ) + b) )
        if(np.sign(ans) == 1):#mistake, this should be negative
            num_mistakes += 1
            weight_vec -= ETA * ( weight_vec / N_TOT - C * -1.0 * training_neg_arr[i] )
        else:
            weight_vec -= ETA * weight_vec / N_TOT
        ans = 1 - ( -1.0 * ( np.dot( np.transpose(weight_vec) , training_neg_arr[i] ) + b) )
        if(np.sign(ans) == 1):
            b += C * ETA * -1.0
    for i in range(N_POS_WITHHELD):
        ans = 1 - ( 1.0 * ( np.dot( np.transpose(weight_vec) , dev_pos_arr[i] ) + b) )
        if(np.sign(ans) == 1):#mistake, this should be negative
            num_mistakes += 1
    for i in range(N_NEG_WITHHELD):
        ans = 1 - ( -1.0 * ( np.dot( np.transpose(weight_vec) , dev_neg_arr[i] ) + b) )
        if(np.sign(ans) == 1):#mistake, this should be negative
            num_mistakes += 1


    nsteps += 1
    ETA = 1/nsteps
    if(num_mistakes == 0):
        print("No mistakes on training data! Number of steps: {}".format(nsteps))
        done = True
    if( nsteps == CUTOFF_NUMBER ):
        print("Reached maximum number of training iterations ({}). Terminating.".format(CUTOFF_NUMBER))
        done = True

        
print("Checking SVM's abilities...")
num_mistakes = 0
false_pos_count = 0
true_pos_count = 0
for i in range(len(dev_pos_arr)):#should be positive
    ans = 1 - ( 1.0 * ( np.dot( np.transpose(weight_vec) , dev_pos_arr[i] ) + b) )
    if(np.sign(ans) == 1):
        num_mistakes += 1
    else:
        true_pos_count += 1
for i in range(len(dev_neg_arr)):#should be positive b/c we use -1.0
    ans = 1 - ( -1.0 * ( np.dot( np.transpose(weight_vec) ,dev_neg_arr[i] ) + b) )
    if(np.sign(ans) == 1):
        num_mistakes += 1
        false_pos_count += 1
print("false positives: ", false_pos_count)
print("true positives: ", true_pos_count)
print("After {} iterations of training, this SVM incorrectly categorized {} out of {} data points from our testing set.".format(nsteps, num_mistakes, len(dev_pos_arr) + len(dev_neg_arr)))

accuracy = (1.0 - float(num_mistakes)/float(len(dev_pos_arr) + len(dev_neg_arr))) * 100
print("Accuracy : {}%\nTPR : {}\nFPR : {}".format(accuracy, true_pos_count/len(dev_pos_arr), false_pos_count/len(dev_neg_arr)))
