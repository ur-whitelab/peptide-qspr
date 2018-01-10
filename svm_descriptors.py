import numpy as np
from random import randint, shuffle
import sys
import pandas as pd
from sklearn import svm
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
import math

ETA = 1.0#the initial learning rate
b = 0.0#The initial b value. Might as well start at 0
C = 0.2#The C value. Small for small changes to b
DATA_DIR = 'data/' #path to data directory

def printHelp():
    print("Usage: svm_descriptors.py [positive file] [negative file] [num_runs] [logfile name] [OPTIONAL: list descriptors to use. Can be one or two of: 'netCharge', 'nChargedGroups', or 'nNonPolarGroups'. Uses all three by default.]")
    exit(1)

if(len(sys.argv) != 5 and len(sys.argv) != 6):
    printHelp()
keys = []
pos_file = sys.argv[1]
neg_file = sys.argv[2]
CUTOFF_NUMBER = int(sys.argv[3])
log_file = sys.argv[4]
if(len(sys.argv) == 6):
    keys.append(sys.argv[5])
elif(len(sys.argv) == 7):
    keys.append(sys.argv[5])
    keys.append(sys.argv[6])
else:
    keys = ['netCharge', 'nChargedGroups', 'nNonPolarGroups']

N_FEATURES = len(keys)

print('Using keys: {}'.format(keys))

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
        for j in range(N_FEATURES):
            train_arr[i][j] = data[keys[j]][i]
    for i in range( (len(data) - end)):
        for j in range(N_FEATURES):
            dev_arr[i][j] = data[keys[j]][i+end]
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
while(not done):
    if(nsteps%50 == 0):
        print("Beginning training step {}".format(nsteps))
    num_train_mistakes = 0
    num_dev_mistakes = 0
    for i in range(N_POS):
        #positive sign b/c we have a positive case
        ans = 1 - ( 1.0 * ( np.dot( np.transpose(weight_vec) , training_pos_arr[i] ) + b) )
        if(np.sign(ans) == 1):#mistake, this should be negative
            num_train_mistakes += 1
            weight_vec -= ETA * ( weight_vec / (N_TOT) - C * 1.0 * training_pos_arr[i] )
        else:
            weight_vec -= ETA * weight_vec / N_TOT
        ans = 1 - ( 1.0 * ( np.dot( np.transpose(weight_vec) , training_pos_arr[i] ) + b) )
        if(np.sign(ans) == 1):
            b += C * ETA * 1.0
    for i in range(N_NEG):#-1 here because these are from the negative set
        ans = 1 - ( -1.0 * ( np.dot( np.transpose(weight_vec) , training_neg_arr[i] ) + b) )
        if(np.sign(ans) == 1):#mistake, this should be negative
            num_train_mistakes += 1
            weight_vec -= ETA * ( weight_vec / N_TOT - C * -1.0 * training_neg_arr[i] )
        else:
            weight_vec -= ETA * weight_vec / N_TOT
        ans = 1 - ( -1.0 * ( np.dot( np.transpose(weight_vec) , training_neg_arr[i] ) + b) )
        if(np.sign(ans) == 1):
            b += C * ETA * -1.0
    #do the same counts for the withheld data, but don't update the vector
    for i in range(N_POS_WITHHELD):
        ans = 1 - ( 1.0 * ( np.dot( np.transpose(weight_vec) , dev_pos_arr[i] ) + b) )
        if(np.sign(ans) == 1):#mistake, this should be negative
            num_dev_mistakes += 1
    for i in range(N_NEG_WITHHELD):
        ans = 1 - ( -1.0 * ( np.dot( np.transpose(weight_vec) , dev_neg_arr[i] ) + b) )
        if(np.sign(ans) == 1):#mistake, this should be negative
            num_dev_mistakes += 1


    nsteps += 1
    ETA = 1/nsteps
    if(num_train_mistakes == 0):
        print("No mistakes on training data! Number of steps: {}".format(nsteps))
        done = True
    if(num_dev_mistakes == 0):
        print("No mistakes on withheld  data! Number of steps: {}".format(nsteps))
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
print("After {} iterations of training, this SVM incorrectly categorized {} out of {} data points from our testing set.".format(nsteps, num_mistakes, (len(training_pos_arr) + len(training_neg_arr) + len(dev_pos_arr) + len(dev_neg_arr)) ))

accuracy = (1.0 - float(num_mistakes)/float(len(training_pos_arr) + len(training_neg_arr) + len(dev_pos_arr) + len(dev_neg_arr))) * 100
print("Accuracy : {}%\nTPR : {}\nFPR : {}".format(accuracy, true_pos_count/(len(training_pos_arr) + len(dev_pos_arr)), false_pos_count/(len(training_neg_arr) + len(dev_neg_arr))))


print('Using scikit learn SVM...')

clf_training_data = np.append(training_pos_arr, training_neg_arr, axis = 0)
clf_training_labels = [1 for item in training_pos_arr]
for item in training_neg_arr:
    clf_training_labels.append(-1)
clf_dev_data = np.append(dev_pos_arr, dev_neg_arr, axis = 0)
clf_dev_labels = [1 for item in dev_pos_arr]
for item in dev_neg_arr:
    clf_dev_labels.append(-1)

def get_best_idx(fpr_arr, tpr_arr, prefactor):
    best_idx = 0
    old_dist = 2
    for j in range(1,len(fpr_arr)):
        dist = math.sqrt(prefactor*fpr_arr[j] **2 + (1-tpr_arr[j]) **2)
        if(old_dist > dist):
            best_idx = j
            old_dist = dist
    return(best_idx)
    
clf = svm.SVC()
clf.fit(clf_training_data, clf_training_labels)
clf_score = clf.score(clf_dev_data, clf_dev_labels)
#np.unique() gives sorted array of unique elems <-- first index is 1 always
#with return_counts=True, gives the counts as second array <-- second count is positives
clf_decision_func = clf.decision_function(clf_dev_data)
clf_fpr, clf_tpr, _ = roc_curve(clf_dev_labels, clf_decision_func, pos_label=1)
best_clf_idx = get_best_idx(clf_fpr, clf_tpr, 2.0)

clf2 = svm.LinearSVC()
clf2.fit(clf_training_data, clf_training_labels)
clf2_score = clf.score(clf_dev_data, clf_dev_labels)
clf2_decision_func = clf2.decision_function(clf_dev_data)
clf2_fpr, clf2_tpr, _ = roc_curve(clf_dev_labels, clf2_decision_func, pos_label=1)
best_clf2_idx = get_best_idx(clf2_fpr, clf2_tpr, 2.0)

print('Using the sklearn C-Support SVM, the average accuracy on the dev sets was: {}'.format(clf_score))

print('Using the sklearn Linear SVM, the average accuracy on the dev sets was: {}'.format(clf2_score))

with open('{}/{}.'.format(DATA_DIR, log_file), 'w+') as f:
    f.write('positives file: {}\n'.format(pos_file))
    f.write('negatives file: {}\n'.format(neg_file))
    f.write("false positives: {}\n".format(false_pos_count))
    f.write("true positives: {}\n".format( true_pos_count))
    f.write("false negatives: {}\n".format( num_mistakes - false_pos_count))
    f.write("true negatives: {}\n".format( (len(dev_neg_arr) + len(training_neg_arr)) - false_pos_count))
    f.write("After {} iterations of training, this SVM incorrectly categorized {} out of {} data points from our testing set.".format(nsteps, num_mistakes, len(dev_pos_arr) + len(dev_neg_arr)))
    f.write("Accuracy : {}%\nTPR : {}\nFPR : {}".format(accuracy, true_pos_count/(len(training_pos_arr) + len(dev_pos_arr)), false_pos_count/(len(training_neg_arr) + len(dev_neg_arr))))
    f.write('Using the sklearn C-Support SVM, the average accuracy on the dev sets was: {}'.format(clf_score))
    f.write('Using the sklearn Linear SVM, the average accuracy on the dev sets was: {}'.format(clf2_score))

plt.rcParams.update({'font.size': 7})
plt.figure(figsize = (2.5, 2.0), dpi=800)

plt.plot([0,1],[0,1], ls=':', label='totally random', color='black')
plt.plot(clf_fpr, clf_tpr, 'o', label='ROC at varied cutoffs', color='red', lw=2.0, ms=2.0)
plt.plot(clf_fpr[best_clf_idx], clf_tpr[best_clf_idx], 's', color='blue', label='best cutoff')
plt.title('ROC Curve for C-Vector SVM')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.savefig('{}/C-Vector_ROC.png'.format(DATA_DIR))
plt.savefig('{}/C-Vector_ROC.svg'.format(DATA_DIR))
plt.savefig('{}/C-Vector_ROC.pdf'.format(DATA_DIR))

plt.figure(figsize = (2.5, 2.0), dpi=800)
plt.plot([0,1],[0,1], ls=':', label='totally random', color='black')
plt.plot(clf2_fpr, clf2_tpr, color='red')
plt.plot(clf2_fpr[best_clf2_idx], clf2_tpr[best_clf2_idx], 's', color='blue', label='best cutoff')
#plt.title('ROC Curve for Linear SVM')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.savefig('{}/Linear_ROC.png'.format(DATA_DIR))
plt.savefig('{}/Linear_ROC.svg'.format(DATA_DIR))
plt.savefig('{}/Linear_ROC.pdf'.format(DATA_DIR))

