import Similarity.GetSimilarity
import Features.CommonFunctions as cF

ListofTopicWords = []
IsWindowBased = False
NumberofSentencesToMatch = 3

def readTopicLists(topicsFilePath):
    global ListofTopicWords
    ListofTopicWords = cF.readTopicLists(topicsFilePath)


def matchWindowWithTopicList(window, topicList, matchingIs):
    numberOfMatchingWords = 0
    words = window.split(' ')
    for word in words:
        matchesTopic = Similarity.GetSimilarity.CalculateSimilarityList(word, topicList, matchingIs)
        if matchesTopic == True:
            numberOfMatchingWords += 1
    return numberOfMatchingWords

def runFeature(responseText, matchingIs):
    numberOfSentences = 0
    if IsWindowBased == True:
        numberOfMatchesPerTopic = [0 for i in range(0, len(ListofTopicWords))]
        MatchesPerTopic = [[] for i in range(0, len(ListofTopicWords))]
        sentences = [x for x in responseText.split('.') if (x.strip() != '' and len(x.strip()) > 1)]
        for sentence in sentences:
            sentenceMatchingCountPerTopic = [0 for i in range(0, len(ListofTopicWords))]
            windows = cF.getWindows(sentence)
            for window in windows:
                for i in range(0, len(ListofTopicWords)):
                    topicList = ListofTopicWords[i]
                    matchineWords = matchWindowWithTopicList(window, topicList, matchingIs)
                    if matchineWords >= cF.NumberOfWordsToMatch:
                        sentenceMatchingCountPerTopic[i] = 1
            for i in range(0, len(sentenceMatchingCountPerTopic)):
                if sentenceMatchingCountPerTopic[i] >= 1:
                    numberOfMatchesPerTopic[i] += 1
                    MatchesPerTopic[i].append(sentence)
                    numberOfSentences += 1
                    break
    else:
        sentences = [x for x in responseText.split('.') if (x.strip() != '' and len(x.strip()) > 1)]
        for sentence in sentences:
            wordsMatched = 0
            sentenceWords = sentence.split(' ')
            for word in sentenceWords:
                for i in range(0, len(ListofTopicWords)):
                    topicList = ListofTopicWords[i]
                    matchesTopic = Similarity.GetSimilarity.CalculateSimilarityList(word, topicList, matchingIs)
                    if matchesTopic == True:
                        wordsMatched += 1
            if wordsMatched >= cF.NumberOfWordsToMatch:
                numberOfSentences += 1
    if numberOfSentences < NumberofSentencesToMatch:
        return True
    else:
        return False

#print('Done')