# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 08:58:33 2017

@author: Chris
"""
import pandas as pd

def main():
	wordList = [' ba', ' bi ', ' bs', ' jr', ' r ', ' sr',
			'abap', 'access', 'adobe', 'alteryx', 'analyst', 'audit',
			'auditor', 'bachelors', 'bank', 'objects', 'clearance',
			'client', 'cloud', 'communicat', 'cpa', 'dart', 'database',
			'decision', 'deep learning', 'ecm', 'erp', 'excel', 'financ',
			'fraud', 'fuel', 'gas', 'google', 'hadoop', 'health', 'java',
			'jr', 'junior', 'junior', 'liason', 'machine', 'masters',
			'mining', 'model', 'office', 'oil', 'oss', 'pivot', 'powerpoint',
			'predict', 'present', 'programming', 'python', 'quality', 'estate',
			'relocate', 'report', 'sap', 'sas', 'science', 'security', 'senior',
			'social', 'spss', 'sql', 'sr', 'ssrs', 'statistic', 'tableau',
			'server', 'text', 'travel', 'unstructured', 'visualization', 'word']
	
	arrayOfDocs = docGrabs(98)
	arrayOfTruth = searchArray(arrayOfDocs, wordList)
	arrayOfTruth.to_csv('searchResults2.csv')
	
# ============= Called Functions ==============================================
	
def searchArray(inText, inList):
	finalArray = pd.DataFrame(inList).T # initializes array based on search items
	for row in inText:
		mark = 0
		markList = []
		for word in inList:
			mark = 1 if checkText(row, word) else 0
			markList.append(mark)
		tempArray = pd.DataFrame(markList).T
		finalArray = pd.concat([finalArray, tempArray], ignore_index = True)
	return finalArray

def checkText(inText, charSearch):
	return(charSearch in inText)

def docGrabs(theLength):
	docs = list(range(theLength))
	docList = []
	for words in docs:
		nums = str(words)
		firstDoc = open(nums+".txt").read() # open text files from 0:theLength
		secondDoc = cleanPunctuation(firstDoc)
		docList.append(secondDoc)
	return docList

def makeString(inText):
	outText = []
	for words in inText:
		outText.append(words)
	return outText

def cleanPunctuation(inFile):
	t = inFile.lower()
	t = t.replace('.', ' ')
	t = t.replace(',', ' ')
	t = t.replace('(', ' ')
	t = t.replace(')', ' ')
	t = t.replace(':', ' ')
	t = t.replace(';', ' ')
	t = t.rstrip('\n')
	t = t.rsplit()
	return t

# =========== call main function ==============================================

main()