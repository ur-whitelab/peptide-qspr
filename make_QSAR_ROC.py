import sys
import numpy as np

def printHelp():
    print("APD_gaussmix.py [APD_QSAR_histograms_file] [job] [fake_data_file] [ROC_plot_filename] [optional:motif_file_to_test]")

    
if(len(sys.argv) != 5 and len(sys.argv) != 6):
    printHelp()
    exit()


infile = sys.argv[1]
job = sys.argv[2]
fakefile = sys.argv[3]
plotname = sys.argv[4]
if(len(sys.argv) == 6):
    motiffile=sys.argv[5]
else:
    motiffile=fakefile


'''Process the data...'''
keys = []#the descriptor headings
counts = []
ranges = []
with open(infile, 'r') as f:
    lines = f.readlines()[2:]
    lastkey = 0
    for i in range(len(lines)):
        if '#' in lines[i]:
            keys.append(lines[i][1:-1])
            lastkey = i
            counts.append(lines[lastkey+1][:-1] + lines[lastkey+2][:-1] + lines[lastkey+3][:-1])
            j=1
            linesum = ''
            while((lastkey+j+3) < len(lines) and'#' not in lines[lastkey+j+3] ):
                linesum += lines[lastkey+j+3][:-1]
                j+=1
            ranges.append(linesum)

counts = [list(map(int, counts[i].split())) for i in range(len(counts))]
ranges = [list(map(float, ranges[i].split())) for i in range(len(ranges))]
counts = np.array(counts)
ranges = np.array(ranges)

'''Now we have the histograms... Time to do the ROC making...'''
