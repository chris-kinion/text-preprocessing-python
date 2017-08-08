# This script makes a series of .txt files by number. This helped in organizing scraped data.

numberList = list(range(1,101))

def makeFile(fName):
    newFile = open(fName+".txt", "w")
    newFile.close()

for each in numberList:
	jerry = str(each)
	makeFile(jerry)

