from sklearn import neighbors
from sklearn import ensemble
import numpy as np
from sklearn.svm import NuSVC, SVC
from nltk.metrics.confusionmatrix import ConfusionMatrix
from sklearn import metrics
import Similarity.GetSimilarity as GS
import gensim.models.word2vec

def confusionMatrix(Y_true, Y_predict):
    return ConfusionMatrix([str(y) for y in Y_true], [str(y) for y in Y_predict])


def printConfusionMatrix(confMatrix):
    confMatrixStr = ''
    for i in range(0, len(confMatrix._confusion)):
        for j in range(0, len(confMatrix._confusion[i])):
            confMatrixStr += str(confMatrix._confusion[i][j]) + '\t'
        confMatrixStr += '\n'
    return confMatrixStr

def tempFunc():
    trainingFeatures = []
    trainingScores = []

    readFeatures = open('test_1_NPE_CON_SPC_WOC_.arff','r')
    for line in readFeatures:
        line = line.replace('\n','').replace('\r','')
        parts = line.split(',')
        trainingFeatures.append([float(x) for x in parts[1:-1]])
        trainingScores.append(float(parts[-1]))

    testFeatures = []
    testScores = []
    readFeatures = open('test_2_NPE_CON_SPC_WOC_.arff','r')
    for line in readFeatures:
        line = line.replace('\n','').replace('\r','')
        parts = line.split(',')
        testFeatures.append([float(x) for x in parts[1:-1]])
        testScores.append(float(parts[-1]))

    y = np.asarray(trainingScores)
    X = np.asarray(trainingFeatures)
    clf = ensemble.RandomForestClassifier(max_depth=5)
    clf.fit(X, y)

    y2 = np.asarray(testScores)
    X2 = np.asarray(testFeatures)
    Z = clf.predict(X2)

    QuadKappa = metrics.cohen_kappa_score(testScores,Z, weights="quadratic")
    currentConMatrix = confusionMatrix(testScores, Z)
    print(printConfusionMatrix(currentConMatrix))
    print(str(QuadKappa))

#Similarity.GetSimilarity.loadVectorsFile('C:\\Personal\\Vectors\\English\\w2vec\\GoogleNews-vectors-negative300.bin', True)
model = gensim.models.Word2Vec.load_word2vec_format('../Data/MVP/AllResponses_Expanded_Skipgram_50.bin', binary=True)
gensim.models.Word2Vec.save_word2vec_format(model,'../Data/MVP/AllResponses_Expanded_Skipgram_50.txt', binary=False)