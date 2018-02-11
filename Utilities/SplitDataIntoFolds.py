import codecs


class fold:
    foldID = None
    trainingData = None
    testData = None

    def __init__(self, _foldID, _trainingData, _testData, _trainingScores, _testScores):
        self.foldID = _foldID
        self.trainingData = _trainingData
        self.testData = _testData
        self.trainingScores = _trainingScores
        self.testScores = _testScores


def separateDataIntoFolds(dataPath, numberofFolds, splittingCritrion):
    allData = []
    readDataFile = codecs.open(dataPath, 'r', 'utf-8')
    for line in readDataFile:
        line = line.replace('\n', '').replace('\r', '').strip()
        if line == '':
            continue
        parts = line.split('\t')
        allData.append([parts[0], parts[1], int(parts[2])])

    folds = {}

    if splittingCritrion == 'fixed':
        numberofSamples = len(allData)
        foldSize = int(numberofSamples/ numberofFolds)
        for i in range(0, numberofFolds):
            trainingData = []
            trainingScores = []
            testData = []
            testScores = []

            for j in range(0, numberofSamples):
                if (i + 1) * foldSize > j >= (i) * foldSize:
                    testData.append(allData[j])
                    testScores.append(allData[j][2])
                else:
                    trainingData.append(allData[j])
                    trainingScores.append(allData[j][2])

            currentfold = fold(i+1, trainingData,testData,trainingScores,testScores)
            folds[i+1] = currentfold

    elif splittingCritrion == 'score':
        for i in range(0, numberofFolds):
            trainingData = []
            trainingScores = []
            testData = []
            testScores = []
            for grade in range(1,5):
                specificSamples = [x for x in allData if x[2] == grade]
                numberofSamples = len(specificSamples)
                foldSize = int(numberofSamples / numberofFolds)
                for j in range(0, numberofSamples):
                    if (i + 1) * foldSize > j >= (i) * foldSize:
                        testData.append(specificSamples[j])
                        testScores.append(specificSamples[j][2])
                    else:
                        trainingData.append(specificSamples[j])
                        trainingScores.append(specificSamples[j][2])

            currentfold = fold(i+1, trainingData,testData,trainingScores,testScores)
            folds[i+1] = currentfold
    return folds

def separateFeaturesIntoFolds(dataList, numberofFolds, splittingCritrion):
    folds = {}

    if splittingCritrion == 'fixed':
        numberofSamples = len(dataList)
        foldSize = int(numberofSamples/ numberofFolds)
        for i in range(0, numberofFolds):
            trainingData = []
            trainingScores = []
            testData = []
            testScores = []

            for j in range(0, numberofSamples):
                if (i + 1) * foldSize > j >= (i) * foldSize:
                    testData.append(dataList[j])
                    testScores.append(dataList[j][-1])
                else:
                    trainingData.append(dataList[j])
                    trainingScores.append(dataList[j][-1])

            currentfold = fold(i+1, trainingData,testData,trainingScores,testScores)
            folds[i+1] = currentfold

    elif splittingCritrion == 'score':
        scores = [min([x[-1] for x in dataList]), max([x[-1] for x in dataList])]
        for i in range(0, numberofFolds):
            trainingData = []
            trainingScores = []
            testData = []
            testScores = []
            for grade in range(scores[0],scores[1]+1):
                specificSamples = [x for x in dataList if x[-1] == grade]
                numberofSamples = len(specificSamples)
                foldSize = int(numberofSamples / numberofFolds)
                for j in range(0, numberofSamples):
                    if (i + 1) * foldSize > j >= (i) * foldSize:
                        testData.append(specificSamples[j])
                        testScores.append(specificSamples[j][-1])
                    else:
                        trainingData.append(specificSamples[j])
                        trainingScores.append(specificSamples[j][-1])

            currentfold = fold(i+1, trainingData,testData,trainingScores,testScores)
            folds[i+1] = currentfold
    return folds