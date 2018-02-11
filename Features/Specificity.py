import Similarity.GetSimilarity
import Features.CommonFunctions as cF

windowSize = 6
ListofExampleWords = []

def readExamplesLists(topicsFilePath):
    global ListofExampleWords
    ListofExampleWords = cF.readExampleLists(topicsFilePath)

def matchWindowWithExample(window, example, matchingIs):
    numberOfMatchingWords = 0
    for word in window:
        for exampleWord in example:
            matchesTopic = Similarity.GetSimilarity.CalculateSimilarity(word, exampleWord, matchingIs)
            if matchesTopic == True:
                example.remove(exampleWord)
                numberOfMatchingWords += 1
                break
    if numberOfMatchingWords >= cF.NumberOfWordsToMatch:
        return True
    else:
        return False

def runFeature(responseText, matchingIs):
    exampleMatching = [[0 for j in range(0, len(ListofExampleWords[i]))] for i in range(0, len(ListofExampleWords))]
    sentences = [x for x in responseText.split('.') if (x.strip() != '' and len(x.strip()) > 1)]
    matchedSentences = [{} for j in range(0, len(ListofExampleWords))]
    for sentence in sentences:
        windows = cF.getWindows(sentence)
        for window in windows:
            for i in range(0, len(ListofExampleWords)):
                currentExampleList = ListofExampleWords[i]
                for j in range(0, len(currentExampleList)):
                    #if exampleMatching[i][j] == 1:
                    #    continue
                    example = list(currentExampleList[j])
                    exampleStr = ''
                    for ExWord in example:
                        exampleStr += ExWord + ' '
                    exampleStr = exampleStr.strip()
                    IsMatchFound = matchWindowWithExample(window, example,matchingIs)
                    if IsMatchFound == True:
                        exampleMatching[i][j] = 1
                        if exampleStr in matchedSentences[i].keys():
                            currentList = matchedSentences[i][exampleStr]
                            if sentence not in currentList:
                                currentList.append(sentence)
                            matchedSentences[i][exampleStr] = currentList
                        else:
                            matchedSentences[i][exampleStr] = [sentence]
                        #if sentence+'_'+exampleStr not in matchedSentences[i]:
                        #    matchedSentences[i].append(sentence+'_' + exampleStr)
    return exampleMatching, matchedSentences

#print('Done')