# Text PreProcessing

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.spatial.distance import cosine

import numpy as np
import pandas as pd
import string
import math

def main():
	aliceText = 'AliceAdventuresWonderland.txt' 
	sherlockText = 'SherlockHolmesExcerpt.txt'
	frankText = 'FrankensteinExcerpt.txt'
	
	aliceName = 'Alice in Wonderland'
	sherlockName = 'Sherlock Holmes'
	frankName = 'Frankenstein'
	
	# Tokenize and apply Porter filter
	myDimensionalize(aliceText, aliceName)
	myDimensionalize(sherlockText, sherlockName)
	myDimensionalize(frankText, frankName)
	
	vectorSpace = makeVectorSpace(aliceName, sherlockName, frankName) # Vector Space with one of each 'word' in all three documents
	
	makeBinaryVectorSpace(vectorSpace, aliceName, sherlockName, frankName) # Binary Vector Space
	rtf = makeRTFVectorSpace(vectorSpace, aliceName, sherlockName, frankName) # Raw Term Frequeny (RTF) Vector Space
	ntf = makeNormalTF(rtf, aliceName, sherlockName, frankName) # Normalized Vector Space
	idf = makeTFIDF(ntf, aliceName, sherlockName, frankName) # TF-IDF Calculations

#	COSINE comparisons	
	cos1 = makeCOScompare(idf, aliceName, sherlockName)
	cos2 = makeCOScompare(idf, aliceName, frankName)
	cos3 = makeCOScompare(idf, sherlockName, frankName)

	cosIs(cos1, aliceName, sherlockName)
	cosIs(cos2, aliceName, frankName)
	cosIs(cos3, sherlockName, frankName)

	

def myDimensionalize(txDoc, txName):
	myToken(txDoc, txName)
	myPorter(txDoc, txName)
	
def myToken(tDoc, tName):
	thisDoc = open(tDoc).read()
	thisToken = word_tokenize(thisDoc) # tokenize
	print('The length of the initial lexicon for ', tName, ' is ', len(thisToken)) # display token info


def myPorter(pDoc, pName):
	# stop word removal, print dimensionality
	porterDoc = open(pDoc).read()
	porterDoc = removeGut(porterDoc) # Remove Gutenberg Introduction
	stop_words=set(stopwords.words("english"))
	words = word_tokenize(porterDoc)
	filtered_sentence=[] #create empty list for filtered sentence go into
	for w in words:
		if w not in stop_words:
			filtered_sentence.append(w)  #this will create a list of non-stopword words in sentence
	print('The length of the ', pName, ' lexicon without stop words is ', len(filtered_sentence))
	makeFile(filtered_sentence, "filtered "+pName) # for rtf and normalization process later

	# apply Porter filter
	ps=PorterStemmer()
	porter_sentence=[]
	reducedPorter=[]
	for x in filtered_sentence:
		porter_sentence.append(ps.stem(x))
	for y in porter_sentence:
		if y not in reducedPorter:
			reducedPorter.append(y)
	print('The length of the lexicon after applying the Porter filter to ', pName, ' is ', len(reducedPorter))
	makeFile(reducedPorter, pName)
 
def makeFile(sentence, fName):
    newFile = open(fName+".txt", "w")
    for eachLine in sentence:
        newFile.write(eachLine + " ") # Added space between words to read later
    newFile.close()

def removeGut(gutFile): # This function removes the Gutenberg introduction
    noGut = gutFile[249:] # Gutenberg introduction is 248 characters
    return noGut
				
def makeVectorSpace(aName, bName, cName): # see myPorter where applied Porter filter
	aList = list(open(aName+".txt"))
	bList = list(open(bName+".txt"))
	cList = list(open(cName+".txt"))

	vsList = aList + bList + cList
	list(set(vsList)) # removes duplicates as set and returns an unordered list
	# print(vsList)
	listFile= "vectorSpace.txt"
	newFile = open(listFile, 'w')
	for eachLine in vsList:
		newFile.write(eachLine)
	newFile.close() 
	print("Creation of vectorSpace.txt successful!")
	return listFile

	aList.close()
	bList.close()
	cList.close()

def fileToList(file):
	aFile = open(file,"r")
	aText=[]
	for each in aFile:
		aText.append(each)

	word_list=[]

	for item in aText:
		for word in item.split():
			word_list.append(word)
	return word_list

	aFile.close()
	
def makeDataFrame(file):
	vSpace = fileToList(file)
	data = {'Words': vSpace}
	myVec = pd.DataFrame(data, columns = ['Words']) # Successful creation of panda dataframe
	return myVec
	
def addBlankCol(dataFrame, colTitle):
	dataFrame[colTitle] = range(len(dataFrame))
	dataFrame.loc[:,colTitle] = np.array([0] * len(dataFrame))
	
def biCompare(dataFrame, mainTitle, compareList, colTitle): # Engine of binary vector creation
	for index, row in dataFrame.iterrows():
		for word in compareList:
			if dataFrame.at[index,mainTitle] == word:
				dataFrame.at[index,colTitle] =  1

def makeBinaryVectorSpace(vsFile, doc1, doc2, doc3):
	vsDF = makeDataFrame(vsFile) # vector space file turned dataframe
	
	biTitle1 = "binary - "+doc1
	biTitle2 = "binary - "+doc2
	biTitle3 = "binary - "+doc3
		
	doc1List = fileToList(doc1+".txt")
	doc2List = fileToList(doc2+".txt")
	doc3List = fileToList(doc2+".txt")
	
	addBlankCol(vsDF, biTitle1)
	addBlankCol(vsDF, biTitle2)
	addBlankCol(vsDF, biTitle3)
	
	biCompare(vsDF, 'Words', doc1List, biTitle1)
	biCompare(vsDF, 'Words', doc2List, biTitle2)
	biCompare(vsDF, 'Words', doc3List, biTitle3)
	print(vsDF)

	print("Completed binary vector space")

def rtfCompare(dataFrame, mainTitle, compareList, colTitle): # Engine of RTF creation
	for index, row in dataFrame.iterrows():
		for word in compareList:
			a = dataFrame.at[index,colTitle]
			if dataFrame.at[index,mainTitle] == word:
				a+=1
				dataFrame.at[index,colTitle] = a

def makeRTFVectorSpace(vsFile, doc1, doc2, doc3):
	vsRTF = makeDataFrame(vsFile)
	
	rtfTitle1= "rtf - "+doc1
	rtfTitle2 = "rtf - "+doc2
	rtfTitle3 = "rtf - "+doc3
		
	doc1List = fileToList("filtered "+doc1+".txt")
	doc2List = fileToList("filtered "+doc2+".txt")
	doc3List = fileToList("filtered "+doc2+".txt")
	
	addBlankCol(vsRTF, rtfTitle1)
	addBlankCol(vsRTF, rtfTitle2)
	addBlankCol(vsRTF, rtfTitle3)

	print("Making raw term frequency columns in dataframe")
	
	rtfCompare(vsRTF, 'Words', doc1List, rtfTitle1)
	rtfCompare(vsRTF, 'Words', doc2List, rtfTitle2)
	rtfCompare(vsRTF, 'Words', doc3List, rtfTitle3)
	
#	print(vsRTF)
	return vsRTF

def compNorm(dFrame, rtfCol, normCol): # Engine of Normalization function
	dFrame[normCol] = dFrame[normCol].astype(np.float)
	normalSum = dFrame[rtfCol].sum()
	
	for index, row in dFrame.iterrows():
		b = dFrame.at[index,rtfCol]
		c = b/normalSum
		dFrame.at[index,normCol] = c
	
def makeNormalTF(makeRTF, doc1, doc2, doc3):
	rtfTitle1= "rtf - "+doc1
	rtfTitle2 = "rtf - "+doc2
	rtfTitle3 = "rtf - "+doc3
	
	normTitle1 = "normal - "+doc1
	normTitle2 = "normal - "+doc2
	normTitle3 = "normal - "+doc3
	
	addBlankCol(makeRTF, normTitle1)
	addBlankCol(makeRTF, normTitle2)
	addBlankCol(makeRTF, normTitle3)
	
	print("Making normalized columns in dataframe")
	
	compNorm(makeRTF, rtfTitle1, normTitle1)
	compNorm(makeRTF, rtfTitle2, normTitle2)
	compNorm(makeRTF, rtfTitle3, normTitle3)

#	print(makeRTF)
	return makeRTF

def makeTFIDF(ntf, doc1, doc2, doc3):	
	rtfTitle1= "rtf - "+doc1
	rtfTitle2 = "rtf - "+doc2
	rtfTitle3 = "rtf - "+doc3
	
	addBlankCol(ntf, "TFIDF")
	addBlankCol(ntf, "sum")
	
	ntf["TFIDF"] = ntf["TFIDF"].astype(np.float)
	print("Making TF-IDF calculations...")

	for index, row in ntf.iterrows():
		a = ntf.at[index,rtfTitle1]
		b = ntf.at[index,rtfTitle2]
		c = ntf.at[index,rtfTitle3]
		rtfSum = a+b+c
		ntf.at[index,"sum"]=rtfSum
		
		thisIDF = math.log(3/(1+rtfSum)) # using constant '3' since only compoaring total of 3 documents
		thisTFIDF = rtfSum * thisIDF
		
		ntf.at[index,"TFIDF"] = thisTFIDF
	
	print(ntf)
	return ntf

def makeCOScompare(tfidf, doc1, doc2): # make cosine comparison of two documents
	normTitle1 = "rtf - "+doc1
	normTitle2 = "rtf - "+doc2
	print("making COSINE comparison between " + doc1 + " and " + doc2)
	
	s = tfidf[normTitle1]
	t = tfidf[normTitle2]
	x = cosine(s,t)

	return x

def cosIs(num, docA, docB):
	print("The cosine of "+docA+" and "+docB+" is ")
	print(num)
	
main()
 
