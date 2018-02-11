import docx2txt
import os
import codecs
import re

def readRater(raterFilePath):
    gradesDic = {}
    readRaterFile = open(raterFilePath,'r')#'G:\PhD\Litman_Work\RTA\Data\Colin\mvp','r')
    for line in readRaterFile:
        line = line.replace('\n','').replace('\r','').strip()
        parts = line.split('\t')
        gradesDic[parts[0].replace('.docx','').replace('.doc','')] = parts[1]
    return gradesDic

def convertDocFilesToSingleCorpora(gradesDic, folderPath, outputPath):
    folderPath = folderPath
    outputPath = outputPath
    writeGradedResponses = codecs.open(outputPath + 'AllResponsesGraded','w','utf-8')
    writeRawResponses = codecs.open(outputPath + 'AllResponses','w','utf-8')

    NonExistindex = 0
    for fileName in os.listdir(folderPath):
        idStr = fileName.replace('.docx','').replace('.doc','')
        try:
            if idStr in gradesDic.keys():

                my_text = docx2txt.process(folderPath + fileName)
                #allLines = my_text.split('\n\r.')#.split('\r').split('.')
                allLines = re.split('\.|\n|\r', my_text)
                writeTextFile = codecs.open(outputPath + idStr+'.txt','w','utf-8')
                gradedText = ''
                for line in allLines:
                    filterline = line.replace('\n',' ').replace('\r',' ').strip()
                    filterline = filterline.replace('\t', ' ')
                    filterline = filterline.replace('  ',' ').replace('  ',' ')
                    filterline = filterline.replace(idStr, '')
                    if filterline.strip() == '' or filterline == idStr or filterline in idStr:
                        continue
                    else:
                        writeTextFile.write(filterline + '.\n')
                        writeRawResponses.write(filterline + '.\n')
                        gradedText += filterline + '. '
                writeTextFile.close()
                if gradedText.strip() != '':
                    writeGradedResponses.write(idStr + '\t' + gradedText.strip() + '\t' + gradesDic[idStr] + '\n')
                writeGradedResponses.flush()
                writeRawResponses.flush()
                print(idStr)
        except:
            NonExistindex += 1
    writeGradedResponses.close()
    writeRawResponses.close()
    print(str(NonExistindex))
    print('Done')


def converttxtFilesToSingleCorpora(folderPath, outputPath):
    folderPath = folderPath
    outputPath = outputPath
    writeGradedResponses = codecs.open(outputPath + 'AllResponsesGraded','w','utf-8')
    writeRawResponses = codecs.open(outputPath + 'AllResponses','w','utf-8')

    NonExistindex = 0
    for fileName in os.listdir(folderPath):
        idStr = fileName.replace('.txt','')
        readtxtFile = codecs.open(folderPath + fileName, 'r', 'utf-8')
        my_text = ''
        for line in readtxtFile:
            my_text += line +' '
        #allLines = my_text.split('\n\r.')#.split('\r').split('.')
        allLines = re.split('\.|\n|\r', my_text)
        writeTextFile = codecs.open(outputPath + idStr+'.txt','w','utf-8')
        gradedText = ''
        for line in allLines:
            filterline = line.replace('\n',' ').replace('\r',' ').strip()
            filterline = filterline.replace('\t', ' ')
            filterline = filterline.replace('  ',' ').replace('  ',' ')
            filterline = filterline.replace(idStr, '')
            if filterline.strip() == '' or filterline == idStr or filterline in idStr:
                continue
            else:
                writeTextFile.write(filterline + '.\n')
                writeRawResponses.write(filterline + '.\n')
                gradedText += filterline + '. '
        writeTextFile.close()
        if gradedText.strip() != '':
            writeGradedResponses.write(idStr + '\t' + gradedText.strip() + '\n')
        writeGradedResponses.flush()
        writeRawResponses.flush()
        print(idStr)
    writeGradedResponses.close()
    writeRawResponses.close()
    print(str(NonExistindex))
    print('Done')

#mvpDic = readRater('G:\\PhD\\Litman_Work\\RTA\\Data\\Colin\\space\\evi.txt')
#convertDocFilesToSingleCorpora(mvpDic,'G:\\PhD\\Litman_Work\\RTA\\Data\\Colin\\Grossman Space Typed Only\\', 'G:\\PhD\\Litman_Work\\RTA\\RTAProject\\Data\\Space\\responses\\')

#mvpDic = readRater('G:\\PhD\\Litman_Work\\RTA\\Data\\Colin\\mvp\\evi.txt')
#convertDocFilesToSingleCorpora(mvpDic,'G:\\PhD\\Litman_Work\\RTA\\Data\\Colin\\MVP All\\', 'G:\\PhD\\Litman_Work\\RTA\\RTAProject\\Data\\MVP\\responses\\')

converttxtFilesToSingleCorpora('G:\\PhD\\Litman_Work\\RTA\\RTAProject\\Data\\NewSpace\\responses\\', 'G:\\PhD\\Litman_Work\\RTA\\RTAProject\\Data\\NewSpace\\')