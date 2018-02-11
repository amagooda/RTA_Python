import codecs
import Features.Specificity
import Features.Concentration
import Features.Wordcount
import Features.NumberEvidences
import Utilities.SplitDataIntoFolds
import Similarity.GetSimilarity as GS
import Similarity
from sklearn import metrics
import pickle
import os
import Sentiment.CalculateSentiment
import Classification.useClassifier
from nltk.metrics.confusionmatrix import ConfusionMatrix
import math
import Features.CommonFunctions as cF

extractNOE = True
extractCon = False
extractSpec = True
extractWordCount = False

def extractFeatures(fileName,_matchingAlgorithm, featureID, useSentiment):
    if os.path.exists('./'+featureID):
        readFeatures = open('./' + featureID, 'rb')
        allData = list(pickle.load(readFeatures))
        readFeatures.close()
        readFeatures = open('./' + featureID+'_Evidences', 'rb')
        allEvidences = list(pickle.load(readFeatures))

        readFeatures = open('./' + featureID+'_Specificity', 'rb')
        allSpecificExamples = list(pickle.load(readFeatures))
        readFeatures.close()


        allFeatureVectors = []
        for y in allData:
            x = y[2]
            featureVector = []
            featureVector.append(float(x[0]))
            featureVector.append(float(x[1]))
            for i in range(2,len(x) - 4):
                featureVector.append(float(x[i]))
            featureVector.append(float(x[-4]))
            if useSentiment == True:
                featureVector.append(x[-3])
                featureVector.append(x[-2])
                featureVector.append(x[-1])
            allFeatureVectors.append([y[0], y[1], featureVector, int(y[-1])])
        return allFeatureVectors, allEvidences, allSpecificExamples
    else:
        GS.clearTempDics()
        allData = []
        allEvidences = []
        allSpecificExamples = []
        readDataFile = codecs.open(fileName, 'r', 'utf-8')
        for line in readDataFile:
            line = line.replace('\n', '').replace('\r', '').strip()
            if line == '':
                continue
            parts = line.lower().split('\t')

            response = parts[1].replace(',', ' ').replace(';',' ').replace('\"',' ').replace('  ',' ').replace('  ',' ')

            currentSampleFeatureVector = []
            if extractNOE == True:
                NOE, Evidences = Features.NumberEvidences.runFeature(response, _matchingAlgorithm)
                allEvidences.append(Evidences)
                NOEFeature = len([x for x in NOE if x > 0])
                currentSampleFeatureVector.append(NOEFeature)

            if extractCon == True:
                CON = Features.Concentration.runFeature(response, _matchingAlgorithm)
                if CON == True:
                    currentSampleFeatureVector.append(1)
                else:
                    currentSampleFeatureVector.append(0)

            if extractSpec == True:
                SPEC, matchSent = Features.Specificity.runFeature(response, _matchingAlgorithm)
                allSpecificExamples.append(matchSent)
                for elem in SPEC:
                    currentSampleFeatureVector.append(sum(elem))

            if extractWordCount == True:
                WCON = Features.Wordcount.runFeature(response)
                currentSampleFeatureVector.append(WCON)

            if useSentiment == True:
                sentVal1 = Sentiment.CalculateSentiment.CalculateSentiment(response)
                sentVal2,sentVal3 = Sentiment.CalculateSentiment.CalculateSentimentVader(response)
                currentSampleFeatureVector.append(sentVal1)
                currentSampleFeatureVector.append(sentVal2)
                currentSampleFeatureVector.append(sentVal3)

            allData.append([parts[0], parts[1], currentSampleFeatureVector, int(parts[2])])
            print('extracting Sample : ' + str(len(allData)) + '\t' + _matchingAlgorithm)
        writeFeatures = open('./' + featureID, 'wb')
        pickle.dump(allData, writeFeatures)
        writeFeatures.close()
        writeFeatures = open('./' + featureID+'_Evidences', 'wb')
        pickle.dump(allEvidences, writeFeatures)
        writeFeatures.close()
        writeFeatures = open('./' + featureID + '_Specificity', 'wb')
        pickle.dump(allSpecificExamples, writeFeatures)
        writeFeatures.close()
    return allData, allEvidences, allSpecificExamples

def extractFeaturesForSentimentClassification(fileName, featureID):
    if os.path.exists('./'+featureID):
        readFeatures = open('./' + featureID, 'rb')
        allData = list(pickle.load(readFeatures))
        readFeatures.close()
        return allData

    else:
        allData = []
        readDataFile = codecs.open(fileName, 'r', 'utf-8')
        for line in readDataFile:
            currentSampleFeatureVector = []
            line = line.replace('\n', '').replace('\r', '').strip()
            if line == '':
                continue
            parts = line.split('\t')
            sentVal1 = Sentiment.CalculateSentiment.CalculateSentiment(parts[1])
            sentVal2,sentVal3 = Sentiment.CalculateSentiment.CalculateSentimentVader(parts[1])

            fileID = parts[0]
            sentimentVal = -1
            sentimentTag = fileID.split('_')[-1]
            if len(sentimentTag) == 1:
                if sentimentTag.lower() == 'p':
                    sentimentVal = 1
                else:
                    sentimentVal = -1
            elif len(sentimentTag) == 2:
                sentimentVal = 0
            elif len(sentimentTag) > 2:
                sentimentVal = 0

            currentSampleFeatureVector.append(sentVal1)
            currentSampleFeatureVector.append(sentVal2)
            currentSampleFeatureVector.append(sentVal3)

            allData.append([fileID, parts[1] ,currentSampleFeatureVector, sentimentVal])
            print('extracting sentiment Sample : ' + str(len(allData)))
        writeFeatures = open('./' + featureID, 'wb')
        pickle.dump(allData, writeFeatures)
        writeFeatures.close()
    return allData

def runOnData(foldsFixed, foldsScore, _matchingAlgorithm, classifier, param1, param2):
    scoreBasedKappa = 0
    ScoreConfusionMatrix = None
    for i in range(1, 11):
        #print(_matchingAlgorithm + ' Score based fold : ' + str(i))
        currentFold = foldsScore[i]
        trainedModel = Classification.useClassifier.trainModel(currentFold.trainingData,currentFold.trainingScores,classifier,param1, param2)
        predictions = Classification.useClassifier.testModel(currentFold.testData,trainedModel)
        if classifier == 'SVR':
            for k in range(0, len(predictions)):
                predictions[k] = int(predictions[k]+0.5)
                if predictions[k] <= 1:
                    predictions[k] = 1
                elif predictions[k] >= 4:
                    predictions[k] = 4

        foldKappa = metrics.cohen_kappa_score(currentFold.testScores,predictions,weights="quadratic")
        scoreBasedKappa += foldKappa

        if ScoreConfusionMatrix == None:
            ScoreConfusionMatrix = confusionMatrix(currentFold.testScores, predictions)
        else:
            currentConMatrix = confusionMatrix(currentFold.testScores, predictions)
            for i in range(0, 4):#len(currentConMatrix._confusion)):
                for j in range(0, 4):#len(currentConMatrix._confusion[i])):
                    #print(str(currentConMatrix._confusion[i][j]))
                    ScoreConfusionMatrix._confusion[i][j] += currentConMatrix._confusion[i][j]
    scoreBasedKappa = float(scoreBasedKappa) / 10.0



    fixedKappa = 0
    # for i in range(1, 11):
    #     #print(_matchingAlgorithm + ' Fixed fold : ' + str(i))
    #     currentFold = foldsFixed[i]
    #     trainedModel = Classification.useClassifier.trainModel(currentFold.trainingData,currentFold.trainingScores,classifier,param1, param2)
    #     predictions = Classification.useClassifier.testModel(currentFold.testData,trainedModel)
    #     foldKappa = metrics.cohen_kappa_score(currentFold.testScores,predictions,weights="quadratic")
    #     fixedKappa += foldKappa
    # fixedKappa = float(fixedKappa) / 10.0

    return scoreBasedKappa, fixedKappa, ScoreConfusionMatrix

def runOnDataSentiment(foldsFixed, foldsScore, classifier, param1, param2):
    scoreBasedKappa = 0
    for i in range(1, 11):
        print('Score based fold : ' + str(i))
        currentFold = foldsScore[i]
        trainedModel = Classification.useClassifier.trainModel(currentFold.trainingData,currentFold.trainingScores,classifier,param1, param2)
        predictions = Classification.useClassifier.testModel(currentFold.testData,trainedModel)
        foldKappa = metrics.f1_score(currentFold.testScores,predictions, average='micro')
        scoreBasedKappa += foldKappa
    scoreBasedKappa = float(scoreBasedKappa) / 10.0

    fixedKappa = 0
    for i in range(1, 11):
        print('Fixed fold : ' + str(i))
        currentFold = foldsFixed[i]
        trainedModel = Classification.useClassifier.trainModel(currentFold.trainingData,currentFold.trainingScores,classifier,param1, param2)
        predictions = Classification.useClassifier.testModel(currentFold.testData,trainedModel)
        foldKappa = metrics.f1_score(currentFold.testScores,predictions, average='micro')
        fixedKappa += foldKappa
    fixedKappa = float(fixedKappa) / 10.0

    return scoreBasedKappa, fixedKappa

def trainSentimentModel(data,classifier, param1, param2):
    trainedModel = Classification.useClassifier.trainModel(data, [x[-1] for x in data],classifier, param1, param2)
    return trainedModel

def confusionMatrix(Y_true , Y_predict):
        return ConfusionMatrix([str(y) for y in Y_true] , [str(y) for y in Y_predict])

def printConfusionMatrix(confMatrix):
    confMatrixStr = '\t   '
    for i in range(0, len(confMatrix._confusion)):
        confMatrixStr += 'P(' + str(i+1)+ ')' + ' '
    confMatrixStr += '\n'

    for i in range(0,len(confMatrix._confusion)):
        confMatrixStr +='H(' + str(i+1)+ ')' + '\t'
        for j in range(0, len(confMatrix._confusion[i])):
            confMatrixStr += str(confMatrix._confusion[i][j]) + '\t'
        confMatrixStr += '\n'
    return confMatrixStr

def runOnDataTwoSteps(sentimentModel, foldsFixed, foldsScore, classifier, param1, param2):
    scoreBasedKappa = 0
    ScoreConfusionMatrix = None

    for i in range(1, 11):
        predictions = []
        print('Score based fold : ' + str(i))
        currentFold = foldsScore[i]
        posData = getPartialData('pos', currentFold.trainingData)
        negData = getPartialData('neg', currentFold.trainingData)
        neuData = getPartialData('neu', currentFold.trainingData)
        posTrainedModel = Classification.useClassifier.trainModel(posData,[x[-1] for x in posData],classifier,param1, param2)
        negTrainedModel = Classification.useClassifier.trainModel(negData,[x[-1] for x in negData], classifier, param1,
                                                                  param2)
        neuTrainedModel = Classification.useClassifier.trainModel(neuData,[x[-1] for x in neuData], classifier, param1,
                                                                  param2)

        for elem in currentFold.testData:
            sentimentFeature = [[elem[0],elem[1],elem[2][-3:]]]
            prediction = Classification.useClassifier.testModel(sentimentFeature, sentimentModel)
            if prediction[0] == -1:
                predictions.append(Classification.useClassifier.testModel([elem], negTrainedModel)[0])
            elif prediction[0] == 0:
                predictions.append(Classification.useClassifier.testModel([elem], neuTrainedModel)[0])
            elif prediction[0] == 1:
                    predictions.append(Classification.useClassifier.testModel([elem],posTrainedModel)[0])
        foldKappa = metrics.cohen_kappa_score(currentFold.testScores,predictions,weights="quadratic")

        if ScoreConfusionMatrix == None:
            ScoreConfusionMatrix = confusionMatrix(currentFold.testScores,predictions)
        else:
            currentConMatrix = confusionMatrix(currentFold.testScores,predictions)
            for i in range(0,len(currentConMatrix._confusion)):
                for j in range(0,len(currentConMatrix._confusion[i])):
                    ScoreConfusionMatrix._confusion[i][j] += currentConMatrix._confusion[i][j]

        scoreBasedKappa += foldKappa
    scoreBasedKappa = float(scoreBasedKappa) / 10.0

    fixedKappa = 0
    # for i in range(1, 11):
    #     print('Fixed fold : ' + str(i))
    #     predictions = []
    #     currentFold = foldsFixed[i]
    #     posData = getPartialData('pos', currentFold.trainingData)
    #     negData = getPartialData('neg', currentFold.trainingData)
    #     neuData = getPartialData('neu', currentFold.trainingData)
    #     posTrainedModel = Classification.useClassifier.trainModel(posData,[x[-1] for x in posData],classifier,param1, param2)
    #     negTrainedModel = Classification.useClassifier.trainModel(negData,[x[-1] for x in negData], classifier, param1,
    #                                                               param2)
    #     neuTrainedModel = Classification.useClassifier.trainModel(neuData,[x[-1] for x in neuData], classifier, param1,
    #                                                               param2)
    #
    #     for elem in currentFold.testData:
    #         sentimentFeature = [[elem[0],elem[1],elem[2][-3:]]]
    #         prediction = Classification.useClassifier.testModel(sentimentFeature, sentimentModel)
    #         if prediction[0] == -1:
    #             predictions.append(Classification.useClassifier.testModel([elem], negTrainedModel)[0])
    #         elif prediction[0] == 0:
    #             predictions.append(Classification.useClassifier.testModel([elem], neuTrainedModel)[0])
    #         elif prediction[0] == 1:
    #                 predictions.append(Classification.useClassifier.testModel([elem],posTrainedModel)[0])
    #     foldKappa = metrics.cohen_kappa_score(currentFold.testScores,predictions,weights="quadratic")
    #     fixedKappa += foldKappa
    # fixedKappa = float(fixedKappa) / 10.0

    return scoreBasedKappa, fixedKappa, ScoreConfusionMatrix

def writeFeatures(featuresToWrite, fileName):
    writeFe = open('./'+fileName,'w')
    for featureVector in featuresToWrite:
        writeFe.write(featureVector[0] + '\t' + str(featureVector[2][0])+
                      '\t' + str(featureVector[2][1]) + '\t' + str(featureVector[2][10])
                      + '\t' + str(featureVector[2][2])
                      + '\t' + str(featureVector[2][3])
                      + '\t' + str(featureVector[2][4])
                      + '\t' + str(featureVector[2][5])
                      + '\t' + str(featureVector[2][6])
                      + '\t' + str(featureVector[2][7])
                      + '\t' + str(featureVector[2][8])
                      + '\t' + str(featureVector[2][9]) + '\n')
    writeFe.close()

def getPartialData(dataType, data):
    partPosOfData = []
    partNegOfData = []
    partNeuOfData = []
    for elem in data:
        fileID = elem[0]
        sentimentTag = fileID.split('_')[-1]
        if len(sentimentTag) == 1:
            if sentimentTag.lower() == 'p':
                partPosOfData.append(elem)
            else:
                partNegOfData.append(elem)
        elif len(sentimentTag) == 2:
            partNeuOfData.append(elem)
        elif len(sentimentTag) > 2:
            partNeuOfData.append(elem)

    if dataType == 'pos':
        return partPosOfData
    elif dataType == 'neg':
        return partNegOfData
    elif dataType == 'neu':
        return partNeuOfData

def runAllCombinations():
    dataSets = ['Space', 'MVP']
    classifiers = ['KNN', 'SVM', 'RF']

    knnParam1 = [5, 11, 17, 23, 29, 37, 55]
    knnParam2 = [1]

    SVMParam1 = [1, 2, 16, 64, 128, 1024, 4096]
    SVMParam2 = [0.04, 0.08, 0.16, 0.32, 0.128, 0.256, 0.512, 1]

    RFParam1 = [20, 50, 80, 100, 150, 200, 300, 500]
    RFParam2 = [3, 5, 8, 10, 15, 20, 30, 40, 50]

    writeResults = open('ClassifierResults', 'w')
    for dataSet in dataSets:
        for classifier in classifiers:
            params1 = []
            params2 = []
            if classifier == 'KNN':
                params1 = knnParam1
                params2 = knnParam2
            elif classifier == 'SVM':
                params1 = SVMParam1
                params2 = SVMParam2
            elif classifier == 'RF':
                params1 = RFParam1
                params2 = RFParam2

            for param1 in params1:
                for param2 in params2:
                    Features.NumberEvidences.readTopicLists(mainFolder + 'topics.txt')
                    Features.Concentration.readTopicLists(mainFolder + 'topics.txt')
                    Features.Specificity.readExamplesLists(mainFolder + 'examples.txt')

                    #Similarity.GetSimilarity.loadVectorsFile(mainFolder + 'AllResponses_Skipgram_50.bin', True)
                    #Similarity.GetSimilarity.loadCCAVectorsFile(mainFolder + 'AllResponses_CCA_50')

                    print(dataSet + '\t' + classifier + '\t' + str(param1) + '\t' + str(param2))
                    mainFolder = './Data/' + dataSet + '/'
                    writeResults.write(classifier + '\t' + str(param1) + '\t' + str(param2) + '\n')
                    useSentiment = True
                    FeaturesExtracted = extractFeatures(mainFolder + 'AllResponsesGraded', 'exact', dataSet + '_exact',
                                                        useSentiment)
                    allFolds = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(FeaturesExtracted, 10, 'score')
                    allFoldsFixed = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(FeaturesExtracted, 10,
                                                                                           'fixed')
                    exactscoreBasedKappa, exactfixedKappa = runOnData(allFoldsFixed, allFolds, 'exact', classifier,
                                                                      param1, param2)

                    Similarity.GetSimilarity.loadVectorsFile(mainFolder + 'AllResponses_Expanded_Skipgram_50.bin', True)
                    FeaturesExtracted = extractFeatures(mainFolder + 'AllResponsesGraded', 'w2vec',
                                                        dataSet + '_w2vec_SG_50_expanded', useSentiment)
                    allFolds = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(FeaturesExtracted, 10, 'score')
                    allFoldsFixed = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(FeaturesExtracted, 10,
                                                                                           'fixed')
                    w2vec_expanded_scoreBasedKappa, w2vec_expanded_fixedKappa = runOnData(allFoldsFixed, allFolds,
                                                                                          'w2vec', classifier, param1,
                                                                                          param2)
                    # -----------------------------------------------------------------------------------------#
                    writeResults.write(dataSet + ' sentiment exact - Score Based : ' + str(exactscoreBasedKappa) + '\n')
                    writeResults.write(dataSet + ' sentiment exact - Fixed Based : ' + str(exactfixedKappa) + '\n')
                    writeResults.write(dataSet + ' sentiment w2vec expanded - Score Based : ' + str(
                        w2vec_expanded_scoreBasedKappa) + '\n')
                    writeResults.write(
                        dataSet + ' sentiment w2vec expanded - Fixed Based : ' + str(w2vec_expanded_fixedKappa) + '\n')



                    useSentiment = False
                    FeaturesExtracted = extractFeatures(mainFolder + 'AllResponsesGraded', 'exact', dataSet + '_exact',
                                                        useSentiment)
                    allFolds = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(FeaturesExtracted, 10, 'score')
                    allFoldsFixed = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(FeaturesExtracted, 10,
                                                                                           'fixed')
                    exactscoreBasedKappa, exactfixedKappa = runOnData(allFoldsFixed, allFolds, 'exact', classifier,
                                                                      param1, param2)

                    Similarity.GetSimilarity.loadVectorsFile(mainFolder + 'AllResponses_Expanded_Skipgram_50.bin', True)
                    FeaturesExtracted = extractFeatures(mainFolder + 'AllResponsesGraded', 'w2vec',
                                                        dataSet + '_w2vec_SG_50_expanded', useSentiment)
                    allFolds = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(FeaturesExtracted, 10, 'score')
                    allFoldsFixed = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(FeaturesExtracted, 10,
                                                                                           'fixed')
                    w2vec_expanded_scoreBasedKappa, w2vec_expanded_fixedKappa = runOnData(allFoldsFixed, allFolds,
                                                                                          'w2vec', classifier, param1,
                                                                                          param2)
                    # -----------------------------------------------------------------------------------------#
                    writeResults.write(dataSet + ' exact - Score Based : ' + str(exactscoreBasedKappa) + '\n')
                    writeResults.write(dataSet + ' exact - Fixed Based : ' + str(exactfixedKappa) + '\n')
                    writeResults.write(
                        dataSet + ' w2vec expanded - Score Based : ' + str(w2vec_expanded_scoreBasedKappa) + '\n')
                    writeResults.write(
                        dataSet + ' w2vec expanded - Fixed Based : ' + str(w2vec_expanded_fixedKappa) + '\n')
                    writeResults.write('-----------------------------------------------------\n')
    writeResults.close()

def writeEvidences(allData, evidences, filePath):
    # writeEvideces = codecs.open(filePath, 'w','utf-8')
    # for x,y in zip(allData,evidences):
    #     studentResponse = x[0] + '\n'#+ x[1]
    #     evidencesInresponses = ''
    #     for evidence in y:
    #         window = evidence[0]
    #         topicWords = evidence[1]
    #         windowString = ''
    #         for word in window:
    #             windowString += word + ' '
    #         topicString = ''
    #         for word in topicWords:
    #             topicString += word + ' '
    #         evidencesInresponses += windowString.strip() + ' | ' + topicString.strip() + '\n'
    #
    #     writeEvideces.write(studentResponse + evidencesInresponses+'\n')
    # writeEvideces.close()
    writeSpec = codecs.open(filePath, 'w','utf-8')
    for x,y in zip(allData,evidences):
        studentResponse = x[0] + '\n'#+ x[1]
        evidencesInresponses = ''
        for specIndex in y:
            if len(specIndex.keys()) == 0:
                evidencesInresponses += '\n'
            else:
                windowString = ''
                for keyStr in specIndex.keys():
                    listofExamples = specIndex[keyStr]
                    windowString += keyStr + '|'
                    for example in listofExamples:
                        windowString += example + ','
                    windowString += '\t'
                evidencesInresponses += windowString + '\n'
        writeSpec.write(studentResponse + evidencesInresponses)
    writeSpec.close()

def writeSpecificity(allData, spec, filePath):
    writeSpec = codecs.open(filePath, 'w','utf-8')
    for x,y in zip(allData,spec):
        studentResponse = x[0] + '\n'#+ x[1]
        evidencesInresponses = ''
        for specIndex in y:
            if len(specIndex.keys()) == 0:
                evidencesInresponses += '\n'
            else:
                windowString = ''
                for keyStr in specIndex.keys():
                    listofExamples = specIndex[keyStr]
                    windowString += keyStr + '|'
                    for example in listofExamples:
                        windowString += example + ','
                    windowString += '\t'
                evidencesInresponses += windowString + '\n'
        writeSpec.write(studentResponse + evidencesInresponses)
    writeSpec.close()

def runAndPrintExperiment(path, MatchingAlgo, featureFile, classifier, param1, param2, useSentiment):
    FeaturesExtracted, evidences = extractFeatures(path, MatchingAlgo, featureFile, useSentiment)
    writeEvidences(FeaturesExtracted, evidences, mainFolder + 'Evidences_'+MatchingAlgo)
    allFolds = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(FeaturesExtracted, 10, 'score')
    allFoldsFixed = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(FeaturesExtracted, 10, 'fixed')
    Kappa, _, confMatrix = runOnData(allFoldsFixed, allFolds, MatchingAlgo, classifier, param1, param2)

    if useSentiment == True:
        print(dataSet + ' ' + MatchingAlgo + ' sentiment , ' + classifier + ' ' + str(param1) +  ' ' + str(param2) + ' : '+ str(Kappa))
    else:
        print(dataSet + ' ' + MatchingAlgo + ' , ' + classifier + ' ' + str(param1) + ' ' + str(param2) + ' : ' + str(Kappa))
    #print(printConfusionMatrix(confMatrix))
    return Kappa

#Similarity.GetSimilarity.loadVectorsFile('C:\\Personal\\Vectors\\English\\w2vec\\GoogleNews-vectors-negative300.bin', True)

# Extract and write features
for dataSet in ['Space','MVP']:
    for matAlgo in ['w2vec_SG_50_expanded']: #'exact',
        mainFolder = './Data/' + dataSet + '/'
        currentPath = mainFolder + 'AllResponsesGraded'
        outputPath = 'output_3\\' + dataSet + '_' + matAlgo +'_'
        Features.NumberEvidences.readTopicLists(mainFolder + 'topics.txt')
        Features.Concentration.readTopicLists(mainFolder + 'topics.txt')
        Features.Specificity.readExamplesLists(mainFolder + 'examples.txt')

        Similarity.GetSimilarity.loadVectorsFile(mainFolder + 'AllResponses_Expanded_Skipgram_50.bin', True)
        FeaturesExtracted, evidences, specificityExamples = extractFeatures(currentPath, matAlgo, outputPath, False)
        writeEvidences(FeaturesExtracted, evidences, outputPath + 'Evidences_' + matAlgo)
        writeSpecificity(FeaturesExtracted, specificityExamples, outputPath + 'Spec_' + matAlgo)

exit()

for dataSet in ['Space','MVP']:
    mainFolder = './Data/' + dataSet + '/'
    Features.NumberEvidences.readTopicLists(mainFolder + 'topics.txt')
    Features.Concentration.readTopicLists(mainFolder + 'topics.txt')
    Features.Specificity.readExamplesLists(mainFolder + 'examples.txt')

    maxKappa = 0
    maxparam1 = 0
    maxparam2 = 0
    maxStride = 0
    writeMaxValue = open(mainFolder + 'MaxAcurracy','w')
    for stride in [1]:#,2,3,4,5]:
        cF.stride = stride
        for param1 in [1,8,16,32]:#,128,1024]:#[10, 20,50,100,200]:#
            for param2 in [0.128, 0.16, 0.32, 1]:#, 4, 8]:#[3, 5,8,10,15]:#

                Similarity.GetSimilarity.loadVectorsFile(mainFolder + 'AllResponses_Expanded_Skipgram_50.bin', True)

                #Similarity.GetSimilarity.loadAndTrainWWNModel('./Similarity/Distances_' + dataSet, dataSet, 'SVM', 32,1)
                #kappa = runAndPrintExperiment(mainFolder + 'AllResponsesGraded', 'WWN',
                                              #'outputs\\' + dataSet + '_WWN_' + str(cF.stride), 'SVR', param1, param2, True)
                #if kappa > maxKappa:
                #    maxKappa = kappa
                #    maxparam1 = param1
                #    maxparam2 = param2
                #    maxStride = stride

                #kappa = runAndPrintExperiment(mainFolder + 'AllResponsesGraded', 'WWN',
                #                              'outputs\\' + dataSet + '_WWN_' + str(cF.stride), 'SVR', param1, param2, False)
                #if kappa > maxKappa:
                #    maxKappa = kappa
                #    maxparam1 = param1
                #    maxparam2 = param2
                #    maxStride = stride


                #Similarity.GetSimilarity.loadAndTrainVWWNModel('./Similarity/VDistances_' + dataSet,dataSet,'SVM', 32,1)
                #kappa = runAndPrintExperiment(mainFolder + 'AllResponsesGraded', 'VWWN', 'outputs\\' + dataSet + '_VWWN_' + str(cF.stride), 'SVR', param1, param2, True)
                #if kappa > maxKappa:
                #    maxKappa = kappa
                #    maxparam1 = param1
                #    maxparam2 = param2
                #    maxStride = stride

                #kappa = runAndPrintExperiment(mainFolder + 'AllResponsesGraded', 'VWWN', 'outputs\\' + dataSet + '_VWWN_' + str(cF.stride), 'SVR', param1, param2, False)
                #if kappa > maxKappa:
                #    maxKappa = kappa
                #    maxparam1 = param1
                #    maxparam2 = param2
                #    maxStride = stride



                kappa = runAndPrintExperiment(mainFolder + 'AllResponsesGraded','exact','outputs\\' + dataSet + '_exact_' + str(cF.stride),'SVR',param1,param2,True)
                if kappa > maxKappa:
                    maxKappa = kappa
                    maxparam1 = param1
                    maxparam2 = param2
                    maxStride = stride
                # kappa = runAndPrintExperiment(mainFolder + 'AllResponsesGraded', 'exact', 'outputs\\' + dataSet + '_exact_' + str(cF.stride), 'SVR', param1, param2, False)
                # if kappa > maxKappa:
                #     maxKappa = kappa
                #     maxparam1 = param1
                #     maxparam2 = param2
                #     maxStride = stride
                #
                # kappa = runAndPrintExperiment(mainFolder + 'AllResponsesGraded','w2vec','outputs\\' + dataSet + '_w2vec_SG_50_expanded_' + str(cF.stride),'SVR',param1,param2,True)
                # if kappa > maxKappa:
                #     maxKappa = kappa
                #     maxparam1 = param1
                #     maxparam2 = param2
                #     maxStride = stride
                #
                # kappa = runAndPrintExperiment(mainFolder + 'AllResponsesGraded', 'w2vec', 'outputs\\' + dataSet + '_w2vec_SG_50_expanded_' + str(cF.stride), 'SVR', param1, param2, False)
                # if kappa > maxKappa:
                #     maxKappa = kappa
                #     maxparam1 = param1
                #     maxparam2 = param2
                #     maxStride = stride

                print('\n')
        print(dataSet  + ' ' + str(maxKappa) + ' ' + str(maxparam1)+ ' ' + str(maxparam2))
        #writeFeatures(FeaturesExtracted, 'features' + dataSet)
    writeMaxValue.write(str(maxStride) + '\t' + str(maxparam1) + '\t' + str(maxparam2) + '\t' + str(maxKappa))
    writeMaxValue.close()
exit()



# Similarity.GetSimilarity.loadVectorsFile(mainFolder + 'AllResponses_Expanded_Skipgram_50.bin', True)
# Similarity.GetSimilarity.loadCCAVectorsFile(mainFolder + 'AllResponses_CCA_50')


# sentimentFeatures = extractFeaturesForSentimentClassification(mainFolder + 'AllResponsesGraded',dataSet + '_senti')
# # allFoldsSentiment = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(sentimentFeatures, 10, 'score')
# # allFoldsSentimentFixed = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(sentimentFeatures, 10, 'fixed')
# # exactscoreBasedF, exactfixedF = runOnDataSentiment(allFoldsSentimentFixed, allFoldsSentiment, 'SVM', 128, 1)
# # print(dataSet + ' exact - Score Based : '+ str(exactscoreBasedF))
# # print(dataSet + ' exact - Fixed Based : '+ str(exactfixedF))
# # exit()
#
# sentimentModel = trainSentimentModel(sentimentFeatures, 'SVM', 128, 1)
#
# FeaturesExtracted = extractFeatures(mainFolder + 'AllResponsesGraded','exact', dataSet+'_exact', True)
# allFolds = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(FeaturesExtracted, 10, 'score')
# allFoldsFixed = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(FeaturesExtracted, 10, 'fixed')
# exact_scoreBasedKappaTwoStepSentiment, _, exact_confMatrixTwoStepSentiment = runOnDataTwoSteps(sentimentModel,allFoldsFixed,allFolds,'RF',50, 15)
#
# FeaturesExtracted = extractFeatures(mainFolder + 'AllResponsesGraded','WWN', dataSet+'_WWN', True)
# allFolds = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(FeaturesExtracted, 10, 'score')
# allFoldsFixed = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(FeaturesExtracted, 10, 'fixed')
# WWN_scoreBasedKappaTwoStepSentiment, _, WWN_confMatrixTwoStepSentiment = runOnDataTwoSteps(sentimentModel,allFoldsFixed,allFolds,'RF',50, 15)
# #
# FeaturesExtracted = extractFeatures(mainFolder + 'AllResponsesGraded', 'w2vec', dataSet + '_w2vec_SG_50_expanded',True)
# allFolds = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(FeaturesExtracted, 10, 'score')
# allFoldsFixed = Utilities.SplitDataIntoFolds.separateFeaturesIntoFolds(FeaturesExtracted, 10,'fixed')
# w2vec_expanded_scoreBasedKappaTwoStepSentiment, _, w2vec_confMatrixTwoStepSentiment = runOnDataTwoSteps(sentimentModel,allFoldsFixed, allFolds, 'RF',50, 15)
# #

# print(dataSet + ' exact sentiment TwoSteps - Score Based : '+ str(exact_scoreBasedKappaTwoStepSentiment))
# print(printConfusionMatrix(exact_confMatrixTwoStepSentiment))
#
# print(dataSet + ' WNN sentiment TwoSteps - Score Based : '+ str(WWN_scoreBasedKappaTwoStepSentiment))
# print(printConfusionMatrix(WWN_confMatrixTwoStepSentiment))
#
# print(dataSet + ' skipgram sentiment TwoSteps - Score Based : '+ str(w2vec_expanded_scoreBasedKappaTwoStepSentiment))
# print(printConfusionMatrix(w2vec_confMatrixTwoStepSentiment))