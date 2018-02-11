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