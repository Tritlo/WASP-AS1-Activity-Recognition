import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statistics as stats
import itertools as it
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
        d = data[i-WINDOW:i]
        stds = np.mean(np.std(d,axis=0))
        rStdDev = np.append(rStdDev, stds)
    return rStdDev

def buildClassifier(runningData,walkingData,standingData, WINDOW=100, STEPS=25):
    calc = lambda data: calcSlidingWindowVals(data, WINDOW, STEPS)
    data = [runningData, walkingData, standingData]
    windows = list(map(calc, data))
    meanVals = list(map(np.mean, windows))
    means = {'running':meanVals[0],
             'walking': meanVals[1],
             'standing': meanVals[2]}
    decisionPoints = []
    for (f,l) in it.combinations(means.items(), 2):
        bySize = sorted([f,l], key=lambda i: i[1])
        p = (bySize[0][1]+bySize[1][1])/2
        n = f'{bySize[0][0].capitalize()}/{bySize[1][0].capitalize()}'
        decisionPoints.append((n,p))
    decisionPoints = sorted(decisionPoints, key=lambda d: d[1])
    return { 'windows': {'running': windows[0],
                         'walking': windows[1],
                         'standing': windows[2]},
             'means': means,
             'decisionPoints': decisionPoints,
             'WINDOW': WINDOW,
             'STEPS': STEPS }

def classify(classifier, data):
    classifiedData = []
    for i in range(classifier['WINDOW'],len(data), classifier['WINDOW']):
        subd = data[i-classifier['WINDOW']:i]
        meanStdDev = np.mean(np.std(subd,axis=0))
        dists = list(map(lambda m: (m[0], abs(meanStdDev - m[1])), classifier['means'].items()))
        (classification,confidence) = min(dists, key=lambda m: m[1])
        for j in range(len(subd)):
            classifiedData.append((classification, subd[j]))
    # Repeate one last time if we missed any at the end.
    # But we need at least 2 elements! Otherwise we just apply the last label.
    if i+2 < len(data):
        subd = data[i:len(data)]
        meanStdDev = np.mean(np.std(subd,axis=0))
        dists = list(map(lambda m: (m[0], abs(meanStdDev - m[1])), classifier['means'].items()))
        (classification,confidence) = min(dists, key=lambda m: m[1])
        for j in range(len(subd)):
            classifiedData.append((classification, subd[j]))
    else:
        for j in range(i,len(data)):
            classifiedData.append((classification,data[j]))
    return np.asarray(classifiedData)


def plotClassifiedData(data):
    labels = data[:,0]
    dataarr = np.asarray(list(map(list,data[:,1])))
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
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.suptitle(f'Classification')
    plt.savefig('Classification.png', dpi=120)





# Plot the classifier and training data
def plotClassifier(classifier):
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
    dp = classifier['decisionPoints']
    l = plt.axhline(dp[-1][1], color='Red', linestyle='dotted')
    l.set_label(dp[-1][0])
    plt.axhspan(dp[-1][1], maxY*1.1, color='Red', alpha=alpha)

    l = plt.axhline(dp[0][1], color='Green', linestyle='dotted')
    l.set_label(dp[0][0])
    plt.axhspan(dp[0][1], dp[-1][1], color='Green', alpha=alpha)

    plt.axhspan(0,dp[0][1], color='Blue', alpha=alpha)

    plt.legend()

    plt.ylabel('Mean Acceleration StDev')
    plt.xlabel('Time')

    #plt.figure(figsize=(12,5))
    plt.suptitle(f'Classifier')
    plt.savefig('Classifier.png', dpi=120)
    plt.close()


running = ".\data\Razan\sensorLog_20200326T183040e5xup4ssiuI_running.txt"
walking = ".\data\Razan\sensorLog_20200326T182913e5xup4ssiuI_walking.txt"
standing = ".\data\Razan\sensorLog_20200326T183205e5xup4ssiuI_standing.txt"
runningAcc = readData(running)
walkingAcc = readData(walking)
standingAcc = readData(standing)
classifier = buildClassifier(runningAcc, walkingAcc, standingAcc)
plotClassifier(classifier)
allData = np.append(standingAcc, np.append(walkingAcc, runningAcc,axis=0),axis=0)
res = classify(classifier,allData)

# Verification
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
print(f'Verification: {str(round(100*correct/len(res),2))}%')
plotClassifiedData(res)



