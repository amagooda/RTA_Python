import gensim.models.word2vec
import numpy
import math
from sklearn import neighbors
from sklearn import ensemble
from sklearn.svm import NuSVC, SVC
import os
import pickle


matchingThresold = 0.7
word2VecModel = None
CCAModel = None
WWNClassificationModel = None

tempWordVecs = {}
tempNormWordVecs = {}

tempWordVecs2 = {}
tempNormWordVecs2 = {}


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / numpy.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return numpy.arccos(numpy.clip(numpy.dot(v1_u, v2_u), -1.0, 1.0)) / 3.14

def distanceBetween(v1, v2):
    dist = 0
    for x,y in zip(v1,v2):
        dist += (x-y)*(x-y)
    dist = math.sqrt(dist)
    return dist

def vectorBetween(vec1,vec2):
    distance = []
    for x, y in zip(vec1,vec2):
        distance.append(x-y)
    return distance

def loadVectorsFile(vectorsFile, isBinary):
    global word2VecModel
    word2VecModel = {}
    if isBinary == False:
        word2VecModel = {}
        readvectorFile = open(vectorsFile,'r')
        index = 0
        for line in readvectorFile:
            index+=1
            if index == 1:
                continue
            parts = line.replace('\n','').replace('\r','').split(' ')
            word2VecModel[parts[0]] = list([float(x) for x in parts[1:]])
        readvectorFile.close()

    else:
        word2VecModel = gensim.models.Word2Vec.load_word2vec_format(vectorsFile, binary=isBinary)

def loadCCAVectorsFile(vectorsFile):
    global CCAModel
    CCAModel = {}
    readVectorsFile = open(vectorsFile,'r')
    for line in readVectorsFile:
        if line.strip() == '':
            continue
        parts = line.split(' ')
        CCAModel[parts[0].strip()] = [float(x) for x in parts[1:]]

def loadAndTrainWWNModel(filePath, dataset, classifier , param1, param2):
    global WWNClassificationModel
    fileName = dataset + '_' + classifier + '_Model_' + str(param1) + '_' + str(param2)
    if os.path.exists('./Similarity/' + fileName):
        readModelFile = open('./Similarity/' + fileName,'rb')
        WWNClassificationModel = pickle.load(readModelFile)
    else:
        trainingFeatures = []
        trainingScores = []
        readFile = open(filePath, 'r')
        for line in readFile:
            if line.replace('\n', '').strip() == '':
                continue
            parts = line.replace('\n', '').strip().split('\t')
            angle = float(parts[0])
            dist = float(parts[1])
            trainingFeatures.append([angle, dist])
            trainingScores.append(int(parts[2]))

        y = numpy.asarray(trainingScores)
        X = numpy.asarray(trainingFeatures)
        if classifier == 'SVM':
            clf = SVC(C=param1, gamma=param2, kernel='rbf')
        elif classifier == 'KNN':
            clf = neighbors.KNeighborsClassifier(param1, weights='distance', algorithm='ball_tree')
        elif classifier == 'RF':
            clf = ensemble.RandomForestClassifier(n_estimators=param1, max_depth=param2)

        clf.fit(X, y)
        WWNClassificationModel = clf

def loadAndTrainVWWNModel(filePath, dataset, classifier , param1, param2):
    global WWNClassificationModel
    fileName = dataset + '_V_' + classifier + '_Model_' + str(param1) + '_' + str(param2)
    if os.path.exists('./Similarity/' + fileName):
        readModelFile = open('./Similarity/' + fileName,'rb')
        WWNClassificationModel = pickle.load(readModelFile)
    else:
        trainingFeatures = []
        trainingScores = []
        readFile = open(filePath, 'r')
        for line in readFile:
            if line.replace('\n', '').strip() == '':
                continue
            parts = line.replace('\n', '').strip().split('\t')
            featureVector = []
            for i in range(0, len(parts) - 1):
                featureVector.append(float(parts[i]))
            trainingFeatures.append(featureVector)
            trainingScores.append(int(parts[-1]))

        y = numpy.asarray(trainingScores)
        X = numpy.asarray(trainingFeatures)
        if classifier == 'SVM':
            clf = SVC(C=param1, gamma=param2, kernel='rbf')
        elif classifier == 'KNN':
            clf = neighbors.KNeighborsClassifier(param1, weights='distance', algorithm='ball_tree')
        elif classifier == 'RF':
            clf = ensemble.RandomForestClassifier(n_estimators=param1, max_depth=param2)

        clf.fit(X, y)
        WWNClassificationModel = clf

def CalculateSimilarity(word, listWord, Algorithm):
    word = word.strip()
    if word == '':
        return False
    if Algorithm == 'exact':
        return exactMatching(word, listWord)
    elif 'w2vec' in Algorithm:# == 'w2vec':
        return w2vecMatching(word, listWord)
    elif Algorithm == 'CCA':
        return CCAMatching(word, listWord)
    elif Algorithm == 'combined':
        return CCA_w2vec_Matching(word, listWord)
    elif Algorithm == 'WWN':
        return w2vec_wordnet_Matching(word, listWord, False)
    elif Algorithm == 'VWWN':
        return w2vec_wordnet_Matching(word, listWord, True)

def CalculateSimilarityList(word, ListofWords, Algorithm):
    word = word.strip()
    if word == '':
        return False
    if Algorithm == 'exact':
        return exactMatchingList(word, ListofWords)
    elif Algorithm == 'w2vec':
        return w2vecMatchingList(word, ListofWords)
    elif Algorithm == 'CCA':
        return CCAMatching(word, ListofWords)
    elif Algorithm == 'combined':
        return CCA_w2vec_Matching(word, ListofWords)
    elif Algorithm == 'WWN':
        return w2vec_wordnet_MatchingList(word, ListofWords, False)
    elif Algorithm == 'VWWN':
        return w2vec_wordnet_MatchingList(word, ListofWords, True)

def exactMatching(word, listWord):
    word = word.strip()
    if (word == listWord) or (word.lower() == listWord):
        return True
    return False

def exactMatchingList(word, listWord):
    word = word.strip()
    if (word in listWord) or (word.lower() in listWord):
        return True
    return False

def w2vecMatching(word, topicWord):
    global word2VecModel

    global tempWordVecs
    global tempNormWordVecs

    global tempWordVecs2
    global tempNormWordVecs2

    wordVector = []
    NormwordVector = None
    topicWordVector = []
    NormtopicWordVecotr = None

    if word in tempWordVecs:
        wordVector = tempWordVecs[word]
        NormwordVector = tempNormWordVecs[word]
    else:
        if word in word2VecModel:
            wordVector = word2VecModel[word]
            NormwordVector = numpy.linalg.norm(wordVector)

            tempWordVecs[word] = wordVector
            tempNormWordVecs[word] = NormwordVector
        else:
            return False

    if topicWord in tempWordVecs2:
        topicWordVector = tempWordVecs2[topicWord]
        NormtopicWordVecotr = tempNormWordVecs2[topicWord]
    else:
        if topicWord in word2VecModel:
            topicWordVector = word2VecModel[topicWord]
            NormtopicWordVecotr = numpy.linalg.norm(topicWordVector)

            tempWordVecs2[topicWord] = topicWordVector
            tempNormWordVecs2[topicWord] = NormtopicWordVecotr
        else:
            return False

    cosine_similarity = numpy.dot(topicWordVector, wordVector)
    cosine_similarity = cosine_similarity / (NormtopicWordVecotr * NormwordVector)

    if cosine_similarity >= matchingThresold:
        return True
    return False

def w2vecMatchingList(word, topicWords):
    global word2VecModel

    global tempWordVecs
    global tempNormWordVecs

    global tempWordVecs2
    global tempNormWordVecs2

    wordVector = []
    NormwordVector = None
    topicWordVector = []
    NormtopicWordVecotr = None

    if word in tempWordVecs:
        wordVector = tempWordVecs[word]
        NormwordVector = tempNormWordVecs[word]
    else:
        if word in word2VecModel:
            wordVector = word2VecModel[word]
            NormwordVector = numpy.linalg.norm(wordVector)

            tempWordVecs[word] = wordVector
            tempNormWordVecs[word] = NormwordVector
        else:
            return False


    for topicWord in topicWords:
        if topicWord in tempWordVecs2:
            topicWordVector = tempWordVecs2[topicWord]
            NormtopicWordVecotr = tempNormWordVecs2[topicWord]
        else:
            if topicWord in word2VecModel:
                topicWordVector = word2VecModel[topicWord]
                NormtopicWordVecotr = numpy.linalg.norm(topicWordVector)

                tempWordVecs2[topicWord] = topicWordVector
                tempNormWordVecs2[topicWord] = NormtopicWordVecotr
            else:
                continue

        cosine_similarity = numpy.dot(topicWordVector, wordVector)
        cosine_similarity = cosine_similarity / (NormtopicWordVecotr * NormwordVector)

        if cosine_similarity >= matchingThresold:
            return True
    return False

def CCAMatching(word, topicList):
    global CCAModel
    wordVector = []
    topicWordVector = []

    if word in CCAModel:
        wordVector = CCAModel[word]
    elif word.lower() in CCAModel:
        word = word.lower()
        wordVector = CCAModel[word]
    else:
        return False
    for topicWord in topicList:
        if topicWord in CCAModel:
            topicWordVector = CCAModel[topicWord]
        elif topicWord.lower() in CCAModel:
            topicWord = topicWord.lower()
            topicWordVector = CCAModel[topicWord]
        else:
            return False

        cosine_similarity = numpy.dot(topicWordVector, wordVector)
        cosine_similarity = cosine_similarity / (numpy.linalg.norm(topicWordVector) * numpy.linalg.norm(wordVector))

        if cosine_similarity >= matchingThresold:
            return True
    return False

def CCA_w2vec_Matching(word, topicList):
    global CCAModel
    global word2VecModel
    CCAWordVector = []
    CCATopicWordVector = []

    w2vecWordVector = []
    w2vecTopicWordVector = []

    if word in CCAModel:
        CCAWordVector = CCAModel[word]
    elif word.lower() in CCAModel:
        word = word.lower()
        CCAWordVector = CCAModel[word]
    else:
        return False
    if word in word2VecModel:
        w2vecWordVector = word2VecModel[word]
    elif word.lower() in word2VecModel:
        word = word.lower()
        w2vecWordVector = word2VecModel[word]
    else:
        return False

    for topicWord in topicList:
        if topicWord in CCAModel:
            CCATopicWordVector = CCAModel[topicWord]
        elif topicWord.lower() in CCAModel:
            topicWord = topicWord.lower()
            CCATopicWordVector = CCAModel[topicWord]
        else:
            return False

        if topicWord in word2VecModel:
            w2vecTopicWordVector = word2VecModel[topicWord]
        elif topicWord.lower() in word2VecModel:
            topicWord = topicWord.lower()
            w2vecTopicWordVector = word2VecModel[topicWord]
        else:
            return False

        CCAcosine_similarity = numpy.dot(CCATopicWordVector, CCAWordVector)
        CCAcosine_similarity /= (numpy.linalg.norm(CCATopicWordVector) * numpy.linalg.norm(CCAWordVector))

        w2veccosine_similarity = numpy.dot(w2vecTopicWordVector, w2vecWordVector)
        w2veccosine_similarity /= (numpy.linalg.norm(w2vecTopicWordVector) * numpy.linalg.norm(w2vecWordVector))

        cosine_similarity = (w2veccosine_similarity + CCAcosine_similarity) / 2.0
        if cosine_similarity >= matchingThresold:
            return True
    return False

def w2vec_wordnet_Matching(word, topicWord, isVector):
    global word2VecModel
    global WWNClassificationModel
    wordVector = []
    topicWordVector = []

    global tempWordVecs
    global tempNormWordVecs

    global tempWordVecs2
    global tempNormWordVecs2

    if word in tempWordVecs:
        wordVector = tempWordVecs[word]
    else:
        if word in word2VecModel:
            wordVector = word2VecModel[word]
            tempWordVecs[word] = wordVector
        else:
            return False

    if topicWord in tempWordVecs2:
        topicWordVector = tempWordVecs2[topicWord]
    else:
        if topicWord in word2VecModel:
            topicWordVector = word2VecModel[topicWord]
            tempWordVecs2[topicWord] = topicWordVector
        else:
            return False

    testFeatures = []
    if isVector == True:
        vec = vectorBetween(topicWordVector,wordVector)
        testFeatures.append(vec)
    else:
        angle = angle_between(topicWordVector, wordVector)
        dist = distanceBetween(topicWordVector, wordVector)
        testFeatures.append([angle,dist])

    Test = numpy.asarray(testFeatures)
    Z = WWNClassificationModel.predict(Test)

    similar = Z[0]
    if similar > 0:
        return True
    else:
        return False

def w2vec_wordnet_MatchingList(word, topicWords, isVector):
    global word2VecModel
    global WWNClassificationModel

    global tempWordVecs
    global tempNormWordVecs

    global tempWordVecs2
    global tempNormWordVecs2

    wordVector = []
    topicWordVector = []

    if word in tempWordVecs:
        wordVector = tempWordVecs[word]
    else:
        if word in word2VecModel:
            wordVector = word2VecModel[word]
            tempWordVecs[word] = wordVector
        else:
            return False

    for topicWord in topicWords:
        if topicWord in tempWordVecs2:
            topicWordVector = tempWordVecs2[topicWord]
        else:
            if topicWord in word2VecModel:
                topicWordVector = word2VecModel[topicWord]
                tempWordVecs2[topicWord] = topicWordVector
            else:
                continue

        testFeatures = []
        if isVector == True:
            vec = vectorBetween(topicWordVector, wordVector)
            testFeatures.append(vec)
        else:
            angle = angle_between(topicWordVector, wordVector)
            dist = distanceBetween(topicWordVector, wordVector)
            testFeatures.append([angle, dist])
        Test = numpy.asarray(testFeatures)
        Z = WWNClassificationModel.predict(Test)

        similar = Z[0]
        if similar > 0:
            return True
    return False

def clearTempDics():
    global tempWordVecs
    global tempNormWordVecs

    global tempWordVecs2
    global tempNormWordVecs2

    tempWordVecs = {}
    tempNormWordVecs = {}

    tempWordVecs2 = {}
    tempNormWordVecs2 = {}