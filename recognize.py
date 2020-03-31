import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statistics as stats
import itertools as it
import os
from collections import Counter

def readData(filename):
    with open(filename) as f:
        acc = []
        for line in f.readlines():
            l = line.split()
            if l[1] == "ACC":
                acc.append(list(map(float, l[2:])))
    return np.asarray(acc)

# We look at WINDOW values at once, and 
# update it after STEPS new measurements
def calcSlidingWindowVals(data, WINDOW, STEPS):
    rStdDev = np.array([])
    for i in range(WINDOW,len(data), STEPS):
        # First d[0:100], then d[25:125], etc.
        # E.g. d =  [[-0.04788403  5.9759274   7.1730285 ]
        #            [-0.05746084  5.995081    7.192182  ]
        #            ...
        #            [ 1.1683705   4.9799395   7.335834  ]
        #            [ 0.9576807   4.8841715   7.0581064 ]]
        d = data[i-WINDOW:i]
        # We compute the means for each axis, i.e.
        # E.g. s = [0.70123355 0.53214879 0.71507838]
        stds = np.std(d,axis=0)
        #  E.g. mstd = 0.6494869049390805
        mstd = np.mean(stds)
        rStdDev = np.append(rStdDev, mstd)
    return rStdDev

def buildClassifier(runningData,walkingData,standingData, WINDOW=100, STEPS=25):
    calc = lambda data: calcSlidingWindowVals(data, WINDOW, STEPS)
    data = [runningData, walkingData, standingData]
    windows = list(map(calc, data))
    meanVals = list(map(np.mean, windows))
    means = {'running':meanVals[0],
             'walking': meanVals[1],
             'standing': meanVals[2]}
    decisionBoundaries = []
    # it.combinations(['w','s','r'],2) = [('w','s'), ('w','r'), ('s','r')]
    for (f,l) in it.combinations(means.items(), 2):
        bySize = sorted([f,l], key=lambda i: i[1])
        p = (bySize[0][1]+bySize[1][1])/2
        n = f'{bySize[0][0].capitalize()}/{bySize[1][0].capitalize()}'
        decisionBoundaries.append((n,p))
    decisionBoundaries = sorted(decisionBoundaries, key=lambda d: d[1])
    return { 'windows': {'running': windows[0],
                         'walking': windows[1],
                         'standing': windows[2]},
             'means': means,
             'decisionBoundaries': decisionBoundaries,
             'WINDOW': WINDOW,
             'STEPS': STEPS }

def classify(classifier, data):
    classifiedData = []
    # Calc calculates the value for the given window,
    #  and adds it to the classified data.
    def classifyWindow(subd):
        meanStdDev = np.mean(np.std(subd,axis=0))
        dists = list(map(lambda m: (m[0], abs(meanStdDev - m[1])), classifier['means'].items()))
        (classification,confidence) = min(dists, key=lambda m: m[1])
        for j in range(len(subd)):
            classifiedData.append((classification, meanStdDev, subd[j]))

    for i in range(classifier['WINDOW'],len(data), classifier['WINDOW']):
        # First data[0:100], then data[100:200], etc.
        window = data[i-classifier['WINDOW']:i]
        classifyWindow(window)
    # Repeate one last time if we missed any at the end.
    # But we need at least 2 elements! Otherwise we just apply the last label.
    if i+2 < len(data):
        window = data[i:len(data)]
        classifyWindow(subd)
    else:
        for j in range(i,len(data)):
            classifiedData.append((classification, meanStdDev, data[j]))
    return np.asarray(classifiedData)


def plotClassifiedData(data, filename="Classification.png"):
    labels = data[:,0]
    means = data[:,1]
    dataarr = np.asarray(list(map(list,data[:,2])))
    maxY = np.max(dataarr)
    minY = np.min(dataarr)

    f = plt.fill_between(range(len(labels)), maxY,minY, where=([l == 'standing' for l in labels]), color='Blue', alpha=0.3 )
    f.set_label('Standing')
    f = plt.fill_between(range(len(labels)), maxY,minY, where=([l == 'walking' for l in labels]), color='Green', alpha=0.3 )
    f.set_label('Walking')
    f = plt.fill_between(range(len(labels)), maxY,minY, where=([l == 'running' for l in labels]), color='Red', alpha=0.3 )
    f.set_label('Running')
    x,y,z = plt.plot(dataarr)
    x.set_label('X Acceleration')
    y.set_label('Y Acceleration')
    z.set_label('Z Acceleration')
    m, = plt.plot(means, linestyle='--',color='yellow')
    m.set_label('Mean StDev')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.suptitle(f'Classification')
    plt.savefig(filename, dpi=120)
    plt.close()





# Plot the classifier and training data
def plotClassifier(classifier, filename):
    plt.figure()
    maxY = max(list(map(np.max, classifier['windows'].values())))

    plt.ylim(0,maxY*1.1)
    line, = plt.plot(classifier['windows']['running'], color='Red')
    plt.axhline(classifier['means']['running'], color='Red', linestyle='--')
    line.set_label('Running')

    line, = plt.plot(classifier['windows']['walking'], color='Green')
    plt.axhline(classifier['means']['walking'], color='Green', linestyle='--')
    line.set_label('Walking')


    line, = plt.plot(classifier['windows']['standing'], color='Blue')
    line.set_label('Standing')
    plt.axhline(classifier['means']['standing'], color='Blue', linestyle='--')




    alpha = 0.3
    db = classifier['decisionBoundaries']
    l = plt.axhline(db[-1][1], color='Red', linestyle='dotted')
    l.set_label(db[-1][0])
    plt.axhspan(db[-1][1], maxY*1.1, color='Red', alpha=alpha)

    l = plt.axhline(db[0][1], color='Green', linestyle='dotted')
    l.set_label(db[0][0])
    plt.axhspan(db[0][1], db[-1][1], color='Green', alpha=alpha)

    plt.axhspan(0,db[0][1], color='Blue', alpha=alpha)

    plt.legend()

    plt.ylabel('Mean Acceleration StDev')
    plt.xlabel('Time')

    #plt.figure(figsize=(12,5))
    plt.suptitle(f'Classifier')
    plt.savefig(filename, dpi=120)
    plt.close()


running = ".\data\Razan\sensorLog_20200326T183040e5xup4ssiuI_running.txt"
walking = ".\data\Razan\sensorLog_20200326T182913e5xup4ssiuI_walking.txt"
standing = ".\data\Razan\sensorLog_20200326T183205e5xup4ssiuI_standing.txt"
runningAcc = readData(running)
walkingAcc = readData(walking)
standingAcc = readData(standing)
print('Building classifier...', end=" ")
classifier = buildClassifier(runningAcc, walkingAcc, standingAcc)
plotClassifier(classifier, "Classifier.png")
print("Done!")

print(f'Computed decision boundaries:')
for k,v in classifier["decisionBoundaries"]:
    print(f'{k}: {v}')


# Verification
allData = np.append(standingAcc, np.append(walkingAcc, runningAcc,axis=0),axis=0)
print(f'Classifying {len(allData)} data points...', end = " ")
res = classify(classifier,allData)
plotClassifiedData(res)
print("Done!")

print("Verifying on training data...", end = " ")
# Ratio calculation
i = 0
labeledTestData = []
for j in range(len(standingAcc)):
    labeledTestData.append(('standing',allData[i]))
    i += 1
for j in range(len(walkingAcc)):
    labeledTestData.append(('walking',allData[i]))
    i += 1
for j in range(len(runningAcc)):
    labeledTestData.append(('running',allData[i]))
    i += 1
labeledTestData = np.asarray(labeledTestData)

correct = 0
for i in range(len(res)):
    if res[i][0] == labeledTestData[i][0]:
        correct += 1

print("Done!")
print(f'Verification: {str(round(100*correct/len(res),2))}% accurate.')

if __name__ == "__main__":
    print("Running!")
    import sys
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        testdir = sys.argv[1]
        print(f'Classifying data in {testdir}')
        testfiles = filter(lambda x: os.path.splitext(x)[1] == ".txt",
                           filter(os.path.isfile,
                                  map(lambda f: os.path.join(testdir,f),
                                      os.listdir(testdir))))
        for datafile in testfiles:
            print(f'Reading data from {datafile}...', end = " ")
            testdata = readData(datafile)
            print('Done!')
            print(f'Classifying...', end = " ")
            testres = classify(classifier,testdata)
            results = list(map(lambda x: x[0], testres))
            print('Done!')
            print(f'Results: {Counter(results)}')
            print(f'Plotting...', end = " ")
            output = os.path.splitext(datafile)[0]
            plotClassifiedData(testres, f'{output}_classified.png')
            print('Done!')

    else:
        print("Usage: python recognize.py <test data directory>")