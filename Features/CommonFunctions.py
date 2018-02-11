windowSize = 6
stride = 5
NumberOfWordsToMatch = 2

def getWindows(sentence):
    ngrams = []
    allwords = [x.strip() for x in sentence.split(' ') if x.strip() != '']

    counter = 0
    gramLength = 0
    currentWord = []

    while gramLength < windowSize and counter < len(allwords):
        currentWord.append(allwords[counter])
        gramLength += 1
        counter += 1
    ngrams.append(list(currentWord))

    while counter < len(allwords) - stride + 1:
        del currentWord[0]
        currentWord.append(allwords[counter])
        ngrams.append(list(currentWord))
        counter += stride
    return ngrams

def readTopicLists(topicsFilePath):
    ListofTopicWords = []
    readLists = open(topicsFilePath)
    for line in readLists:
        line = line.replace('\n','').replace('\r','').strip()
        parts = line.split(',')
        words = [x.strip() for x in parts if x.strip() != '']
        ListofTopicWords.append(words)
    return ListofTopicWords

def readExampleLists(topicsFilePath):
    ListofTopicWords = []
    readLists = open(topicsFilePath)
    for line in readLists:
        line = line.replace('\n', '').replace('\r', '').strip()
        parts = line.split(',')
        examples = [x.strip() for x in parts if x.strip() != '']
        examplesList = []
        for example in examples:
            exampleWords = example.split(' ')
            exampleWords = [x.strip() for x in exampleWords]
            examplesList.append(exampleWords)
        ListofTopicWords.append(examplesList)
    return ListofTopicWords
