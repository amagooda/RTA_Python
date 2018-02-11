import Similarity.GetSimilarity
import Features.CommonFunctions as cF

ListofTopicWords = []

def readTopicLists(topicsFilePath):
    global ListofTopicWords
    ListofTopicWords = cF.readTopicLists(topicsFilePath)


def matchWindowWithTopicList(window, topicList, matchingIs):
    numberOfMatchingWords = 0
    tempTopicList = [x for x in topicList]
    for word in window:
        for topicWord in tempTopicList:
            matchesTopic = Similarity.GetSimilarity.CalculateSimilarity(word, topicWord, matchingIs)
            if matchesTopic == True:
                tempTopicList.remove(topicWord)
                numberOfMatchingWords += 1
                break
    return numberOfMatchingWords

def runFeature(responseText, matchingIs):
    numberOfMatchesPerTopic = [0 for i in range(0, len(ListofTopicWords))]
    #MatchesPerTopic = [[] for i in range(0, len(ListofTopicWords))]
    MatchesPerTopic = [{} for i in range(0, len(ListofTopicWords))]
    sentences = [x for x in responseText.split('.') if (x.strip() != '' and len(x.strip()) > 1)]
    for sentence in sentences:
        sentenceMatchingCountPerTopic = [0 for i in range(0, len(ListofTopicWords))]
        windows = cF.getWindows(sentence)
        for window in windows:
            for i in range(0, len(ListofTopicWords)):
                #if numberOfMatchesPerTopic[i] >= 1:
                #    continue
                topicList = ListofTopicWords[i]
                matchineWords = matchWindowWithTopicList(window, topicList,matchingIs)
                if matchineWords >= cF.NumberOfWordsToMatch:
                    sentenceMatchingCountPerTopic[i] = 1
                    windowString = ''
                    topicString = ''
                    for word in window:
                        windowString += word + ' '
                    windowString = windowString.strip()
                    for topicWord in topicList:
                        topicString += topicWord + ' '
                    topicString = topicString.strip()
                    if topicString in MatchesPerTopic[i].keys():
                        currentTopicsAdded = MatchesPerTopic[i][topicString]
                        if windowString not in currentTopicsAdded:
                            currentTopicsAdded.append(windowString)
                        MatchesPerTopic[i][topicString] = currentTopicsAdded
                    else:
                        MatchesPerTopic[i][topicString] = [windowString]

        for i in range(0, len(sentenceMatchingCountPerTopic)):
            if sentenceMatchingCountPerTopic[i] == 1:
                numberOfMatchesPerTopic[i] += 1
                #MatchesPerTopic[i].append(sentence)
    return numberOfMatchesPerTopic, MatchesPerTopic

#print('Done')