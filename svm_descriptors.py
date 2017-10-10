import numpy as np
from random import randint, shuffle
import sys
import pandas as pd

ETA = 1.0#the initial learning rate
b = 0.0#The initial b value. Might as well start at 0
C = 0.2#The C value. Small for small changes to b
N_FEATURES = 3 # the number of peptide descriptors we're using. Actually hard-code some stuff anyway
DATA_DIR = 'data/' #path to data directory

def printHelp():
    print("Usage: svm_descriptors.py [positive file] [negative file] [num_runs] [logfile name] ")
    exit(1)

if(len(sys.argv) != 5):
    printHelp()

pos_file = sys.argv[1]
neg_file = sys.argv[2]
CUTOFF_NUMBER = sys.argv[3]
log_file = sys.argv[4]


def read_data(filename):
    '''Reads a specified file of the correct input format into target_arr for use in training.
        We only have positive examples in our training set, the APD, so all y-values are 1.0'''
    print("Reading training data from file {}...".format(filename))
    data = pd.read_csv(filename)
    data = data.sample(frac=1).reset_index(drop=True)#randomize so we don't pick same start configs
    end = int((0.8 * len(data)))# the end of the train data/start of test data
    train_arr = np.zeros((end, N_FEATURES))
    dev_arr = np.zeros(((len(data) - end ), N_FEATURES))
    for i in range(end):
        #netCharge, nChargedGroups, nNonPolarGroups are the 3 we care about
        train_arr[i][0] = data['netCharge'][i]
        train_arr[i][1] = data['nChargedGroups'][i]
        train_arr[i][2] = data['nNonPolarGroups'][i]
    for j in range(len(data) - end):
        dev_arr[j][0] = data['netCharge'][i]
        dev_arr[j][1] = data['nChargedGroups'][i]
        dev_arr[j][2] = data['nNonPolarGroups'][i]
    return(train_arr, dev_arr)
    #print(test_arr, dev_arr)
            

print("Setting up...")
#array sizes entered manually, taken from `wc -l <file>` on the respective files
training_pos_arr, dev_pos_arr = read_data(pos_file) 
training_neg_arr, dev_neg_arr = read_data(neg_file)

N_POS = len(training_pos_arr)
N_NEG = len(training_neg_arr)
N_TOT = N_POS + N_NEG
N_POS_WITHHELD = len(dev_pos_arr)
N_NEG_WITHHELD = len(dev_neg_arr)

weight_vec = np.zeros(N_FEATURES)#the w vector, full of random floats 0 to 1

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
for i in range(len(training_pos_arr)):#should be positive
    ans = 1 - ( 1.0 * ( np.dot( np.transpose(weight_vec) , training_pos_arr[i] ) + b) )
    if(np.sign(ans) == 1):
        num_mistakes += 1
    else:
        true_pos_count += 1

for i in range(len(dev_neg_arr)):#should be positive b/c we use -1.0
    ans = 1 - ( -1.0 * ( np.dot( np.transpose(weight_vec) ,dev_neg_arr[i] ) + b) )
    if(np.sign(ans) == 1):
        num_mistakes += 1
        false_pos_count += 1
for i in range(len(training_neg_arr)):#should be positive b/c we use -1.0
    ans = 1 - ( -1.0 * ( np.dot( np.transpose(weight_vec) ,training_neg_arr[i] ) + b) )
    if(np.sign(ans) == 1):
        num_mistakes += 1
        false_pos_count += 1

print("false positives: ", false_pos_count)
print("true positives: ", true_pos_count)
print("false negatives: ", num_mistakes - false_pos_count)
print("true negatives: ", (len(dev_neg_arr) + len(training_neg_arr)) - false_pos_count)
print("After {} iterations of training, this SVM incorrectly categorized {} out of {} data points from our testing set.".format(nsteps, num_mistakes, len(training_pos_arr) + len(training_neg_arr) + len(dev_pos_arr) + len(dev_neg_arr)))

accuracy = (1.0 - float(num_mistakes)/float(len(training_pos_arr) + len(training_neg_arr) + len(dev_pos_arr) + len(dev_neg_arr))) * 100
print("Accuracy : {}%\nTPR : {}\nFPR : {}".format(accuracy, true_pos_count/(len(training_pos_arr) + len(dev_pos_arr)), false_pos_count/(len(training_neg_arr) + len(dev_neg_arr))))

with open(log_file, 'w+') as f:
    f.write('positives file: {}\n'.format(pos_file))
    f.write('negatives file: {}\n'.format(neg_file))
    f.write("false positives: {}".format(false_pos_count))
    f.write("true positives: {}".format( true_pos_count))
    f.write("false negatives: {}".format( num_mistakes - false_pos_count))
    f.write("true negatives: {}".format( (len(dev_neg_arr) + len(training_neg_arr)) - false_pos_count))
    f.write("After {} iterations of training, this SVM incorrectly categorized {} out of {} data points from our testing set.".format(nsteps, num_mistakes, len(dev_pos_arr) + len(dev_neg_arr)))
    f.write("Accuracy : {}%\nTPR : {}\nFPR : {}".format(accuracy, true_pos_count/(len(training_pos_arr) + len(dev_pos_arr)), false_pos_count/(len(training_neg_arr) + len(dev_neg_arr))))

