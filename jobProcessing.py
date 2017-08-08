# Text PreProcessing

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.spatial.distance import cosine

import numpy as np
import pandas as pd
import math

def main():
	indeedText = 'Indeed.txt' 
	usaaText = 'USAA.txt'

	indeedName = 'Indeed'
	usaaName = 'USAA'
	
	# Tokenize and apply Porter filter
	myDimensionalize(indeedText, indeedName)
	myDimensionalize(usaaText, usaaName)
		
	vectorSpace = makeVectorSpace(indeedName, usaaName) # Vector Space with one of each 'word' in both documents
	
	makeBinaryVectorSpace(vectorSpace, indeedName, usaaName) # Binary Vector Space
	rtf = makeRTFVectorSpace(vectorSpace, indeedName, usaaName) # Raw Term Frequeny (RTF) Vector Space
	ntf = makeNormalTF(rtf, indeedName, usaaName) # Normalized Vector Space
	idf = makeTFIDF(ntf, indeedName, usaaName) # TF-IDF Calculations

#	COSINE comparisons	
	cos1 = makeCOScompare(idf, indeedName, usaaName)

	cosIs(cos1, indeedName, usaaName)

# ============= Called Functions ==============================================

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

def makeVectorSpace(aName, bName): # see myPorter where applied Porter filter
	aList = list(open(aName+".txt"))
	bList = list(open(bName+".txt"))

	vsList = aList + bList
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

def makeBinaryVectorSpace(vsFile, doc1, doc2):
	vsDF = makeDataFrame(vsFile) # vector space file turned dataframe
	
	biTitle1 = "binary - "+doc1
	biTitle2 = "binary - "+doc2
		
	doc1List = fileToList(doc1+".txt")
	doc2List = fileToList(doc2+".txt")
	
	addBlankCol(vsDF, biTitle1)
	addBlankCol(vsDF, biTitle2)
	
	biCompare(vsDF, 'Words', doc1List, biTitle1)
	biCompare(vsDF, 'Words', doc2List, biTitle2)
	print(vsDF)

	print("Completed binary vector space")

def rtfCompare(dataFrame, mainTitle, compareList, colTitle): # Engine of RTF creation
	for index, row in dataFrame.iterrows():
		for word in compareList:
			a = dataFrame.at[index,colTitle]
			if dataFrame.at[index,mainTitle] == word:
				a+=1
				dataFrame.at[index,colTitle] = a

def makeRTFVectorSpace(vsFile, doc1, doc2):
	vsRTF = makeDataFrame(vsFile)
	
	rtfTitle1= "rtf - "+doc1
	rtfTitle2 = "rtf - "+doc2
		
	doc1List = fileToList("filtered "+doc1+".txt")
	doc2List = fileToList("filtered "+doc2+".txt")

	addBlankCol(vsRTF, rtfTitle1)
	addBlankCol(vsRTF, rtfTitle2)

	print("Making raw term frequency columns in dataframe")
	
	rtfCompare(vsRTF, 'Words', doc1List, rtfTitle1)
	rtfCompare(vsRTF, 'Words', doc2List, rtfTitle2)
	
	return vsRTF

def compNorm(dFrame, rtfCol, normCol): # Engine of Normalization function
	dFrame[normCol] = dFrame[normCol].astype(np.float)
	normalSum = dFrame[rtfCol].sum()
	
	for index, row in dFrame.iterrows():
		b = dFrame.at[index,rtfCol]
		c = b/normalSum
		dFrame.at[index,normCol] = c
	
def makeNormalTF(makeRTF, doc1, doc2):
	rtfTitle1= "rtf - "+doc1
	rtfTitle2 = "rtf - "+doc2
	
	normTitle1 = "normal - "+doc1
	normTitle2 = "normal - "+doc2
	
	addBlankCol(makeRTF, normTitle1)
	addBlankCol(makeRTF, normTitle2)
	
	print("Making normalized columns in dataframe")
	
	compNorm(makeRTF, rtfTitle1, normTitle1)
	compNorm(makeRTF, rtfTitle2, normTitle2)

	return makeRTF

def makeTFIDF(ntf, doc1, doc2):	
	rtfTitle1= "rtf - "+doc1
	rtfTitle2 = "rtf - "+doc2
	
	addBlankCol(ntf, "TFIDF")
	addBlankCol(ntf, "sum")
	
	ntf["TFIDF"] = ntf["TFIDF"].astype(np.float)
	print("Making TF-IDF calculations...")

	for index, row in ntf.iterrows():
		a = ntf.at[index,rtfTitle1]
		b = ntf.at[index,rtfTitle2]
		rtfSum = a+b
		ntf.at[index,"sum"]=rtfSum
		
		thisIDF = math.log(2/(1+rtfSum)) # using constant '2' since only comparing total of 2 documents
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

# =========== call main function ==============================================

main()
 
