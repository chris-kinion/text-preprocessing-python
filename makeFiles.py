# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 23:50:50 2017

@author: Chris
"""

numberList = list(range(301,303))

def makeFile(fName):
    newFile = open(fName+".txt", "w")
    newFile.close()

for each in numberList:
	jerry = str(each)
	makeFile(jerry)

