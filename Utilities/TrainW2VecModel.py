import codecs

import gensim.models.word2vec
#import numpy
#from gensim.models.keyedvectors import KeyedVectors



def getSentenceVector(sentence, Model, dimensions):
    finalVector = [0 for x in range(0,dimensions)]
    if sentence is not []:
        words = sentence.split(" ")
    else:
        words = sentence

    for word in words:
        if word in Model:
            wordVec = Model[word]
            for i in range(0,len(finalVector)):
                finalVector[i] += wordVec[i]
        else:
            for i in range(0,len(finalVector)):
                finalVector[i] += 0.0000000000001

    return finalVector


def getSentenceVectorArray(sentence, Model,dimensions):
    finalVector = [0 for x in range(0,dimensions)]
    words = sentence

    for word in words:
        if word in Model:
            wordVec = Model[word]
            for i in range(0,len(finalVector)):
                finalVector[i] += wordVec[i]
        else:
            for i in range(0,len(finalVector)):
                finalVector[i] += 0.0000000000001

    return finalVector


def isEqual(sentence1, sentence2):
    if len(sentence1) != len(sentence2):
        return False
    for i in range(0, len(sentence1)):
        if sentence1[i] != sentence2[i]:
            return False
    return True



def trainModel(trainPath, dim, isSkip):
    dimensions = dim
    skipGram = isSkip
    allSentences = []
    readTrainingCorpora = codecs.open(trainPath,'r','utf-8')
    for line in readTrainingCorpora:
        line = line.replace('\n','').replace('\r','').replace('\t',' ').strip()
        if line == '':
            continue
        allSentences.append(line.split(' '))
    readTrainingCorpora.close()

    Model = gensim.models.Word2Vec(allSentences, size=dimensions, window=6, min_count=0, workers=4, hs=1,
                                   sg=skipGram, negative=5)

    return Model

def saveModel(model, path, isBinary):
    #model.save(path,binary=isBinary)
    gensim.models.Word2Vec.save_word2vec_format(model,fname=path, binary=isBinary)

def loadModel(path, isBinary):
    word2VecModel = gensim.models.Word2Vec.load_word2vec_format(path, binary=isBinary)
    return word2VecModel

print('Training MVP_AllResponses_Expanded_skip')
model = trainModel('../Data/MVP/AllResponses_Expanded', 50, True)
saveModel(model, '../Data/MVP/AllResponses_Expanded_Skipgram_50.bin', True)
del model
print('Training MVP_AllResponses_Expanded_CBOW')
model = trainModel('../Data/MVP/AllResponses_Expanded', 50, False)
saveModel(model, '../Data/MVP/AllResponses_Expanded_CBOW_50.bin', True)
del model


print('Training Space_AllResponses_Expanded_Skip')
model = trainModel('../Data/Space/AllResponses_Expanded', 50, True)
saveModel(model, '../Data/Space/AllResponses_Expanded_Skipgram_50.bin', True)
del model
print('Training Space_AllResponses_Expanded_CBOW')
model = trainModel('../Data/Space/AllResponses_Expanded', 50, False)
saveModel(model, '../Data/Space/AllResponses_Expanded_CBOW_50.bin', True)
del model


print('Training Space_AllResponses_Skip')
model = trainModel('../Data/MVP/AllResponses', 50, True)
saveModel(model, '../Data/MVP/AllResponses_Skipgram_50.bin', True)
del model
print('Training Space_AllResponses_CBOW')
model = trainModel('../Data/MVP/AllResponses', 50, False)
saveModel(model, '../Data/MVP/AllResponses_CBOW_50.bin', True)
del model


model = trainModel('../Data/Space/AllResponses', 50, True)
saveModel(model, '../Data/Space/AllResponses_Skipgram_50.bin', True)
del model
model = trainModel('../Data/Space/AllResponses', 50, False)
saveModel(model, '../Data/Space/AllResponses_CBOW_50.bin', True)
del model