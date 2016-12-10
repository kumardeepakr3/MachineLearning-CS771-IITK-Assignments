from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import os, sys, string
import numpy as np
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer


pathToSpamFolder = "/home/deepak/Acads/Sem8/MLT/Assignments/Assgn2/Data/bare/"
subfolderList = ['part1/', 'part2/','part3/','part4/','part5/','part6/','part7/','part8/','part9/','part10/']
stemmer = PorterStemmer()


questionPart = 'A' # Set as A, B, C for the part of the question
# questionPart = 'B'
# questionPart = 'C'

# classifier = 'Perceptron'
classifier = 'LDA'

if(classifier == 'LDA'):
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
elif(classifier == 'Perceptron'):
	from sklearn.linear_model import Perceptron


#Taken from http://stackoverflow.com/questions/26126442/combining-text-stemming-and-removal-of-punctuation-in-nltk-and-scikit-learn
def stem_tokens(tokens, stemmer):
	stemmed = []
	for item in tokens:
		stemmed.append(stemmer.stem(item))
	return stemmed

#Taken from http://stackoverflow.com/questions/26126442/combining-text-stemming-and-removal-of-punctuation-in-nltk-and-scikit-learn
def tokenize(text):
	text = "".join([ch for ch in text if ch not in string.punctuation])
	tokens = word_tokenize(text)
	stems = stem_tokens(tokens, stemmer)
	return stems

def getFilesInFolder(pathOfFolder):
	allFileFullPathList = []
	labelList = []
	for file in os.listdir(pathOfFolder):
		allFileFullPathList.append(pathOfFolder+file)
		#print file
		if file.startswith("spm"):
			labelList.append(1)  # Spam
		else:
			labelList.append(0)  # NonSpam
	return (allFileFullPathList, labelList)

def getFilesInAllFolders(pathOfParent, listOfChildNames):
	combinedFilePaths = []
	combinedLabelList = []
	for childFolder in listOfChildNames:
		(allFileFullPathList, labelList) = getFilesInFolder(pathOfParent+childFolder)
		combinedFilePaths += allFileFullPathList
		combinedLabelList += labelList
	return (combinedFilePaths, combinedLabelList)



for item1 in subfolderList:
	newSubfolderList = []
	for item2 in subfolderList:
		if (item2==item1):
			testFolder = item2
		else:
			newSubfolderList.append(item2)
	#print newSubfolderList
	print "Test Set as: "+ testFolder
	sys.stderr.write(testFolder)
	
	(combinedFilePaths, combinedLabelList) = getFilesInAllFolders(pathToSpamFolder, newSubfolderList)
	#pauseKey = raw_input()

	if(questionPart == 'A'):
		vectorizer = CountVectorizer(input='filename', stop_words='english', tokenizer=tokenize, binary=True)		# PART A
	elif(questionPart == 'B'):
		vectorizer = TfidfVectorizer(input='filename', stop_words='english', tokenizer=tokenize, use_idf=False)	# PART B
	elif(questionPart == 'C'):
		vectorizer = TfidfVectorizer(input='filename', stop_words='english', tokenizer=tokenize)					# PART C
	
	X = vectorizer.fit_transform(combinedFilePaths).toarray()
	#print "vectorizer fittingTransform done"

	if(classifier == 'Perceptron'):
		clf = Perceptron()
	elif(classifier == 'LDA'):
		clf = LinearDiscriminantAnalysis()

	clf.fit(X, np.asarray(combinedLabelList))

	(testFilesPathList, testFilesLabelList) = getFilesInFolder(pathToSpamFolder + testFolder)
	testX = vectorizer.transform(testFilesPathList).toarray()

	answer = clf.score(testX, np.asarray(testFilesLabelList))
	print answer*100
	sys.stderr.write(":" + str(answer*100) + "\n")





# print "NOW FOR ONLY LEMMATIZED, DEFAULTTFIDF"
# sys.stderr.write("NOW FOR ONLY LEMMATIZED, DEFAULTTFIDF")

# for item1 in subfolderList:
# 	newSubfolderList = []
# 	for item2 in subfolderList:
# 		if (item2==item1):
# 			testFolder = item2
# 		else:
# 			newSubfolderList.append(item2)
# 	#print newSubfolderList
# 	print "Test Set as: "+ testFolder
# 	sys.stderr.write(testFolder)
	
# 	(combinedFilePaths, combinedLabelList) = getFilesInAllFolders(pathToSpamFolder, newSubfolderList)
# 	#print combinedFilePaths
# 	#pauseKey = raw_input()
# 	vectorizer = TfidfVectorizer(input='filename', stop_words='english', tokenizer=tokenize)
# 	X = vectorizer.fit_transform(combinedFilePaths).toarray()

# 	#print "vectorizer fittingTransform done"
# 	clf = LinearDiscriminantAnalysis()
# 	clf.fit(X, np.asarray(combinedLabelList))

# 	(testFilesPathList, testFilesLabelList) = getFilesInFolder(pathToSpamFolder + testFolder)
# 	testX = vectorizer.transform(testFilesPathList).toarray()
# 	#testVectorizer = CountVectorizer(input='filename',vocabulary=vectorizer.vocabulary_)
# 	#testX = testVectorizer.fit_transform(testFilesPathList).toarray()
# 	answer = clf.score(testX, np.asarray(testFilesLabelList))

# 	print answer*100
# 	sys.stderr.write(":" + str(answer*100) + "\n")
# 	print "\n"


# print "NOW FOR ONLY LEMMATIZED, DEFAULTTFIDF, BINARYTRUE"
# sys.stderr.write("NOW FOR ONLY LEMMATIZED, DEFAULTTFIDF, BINARYTRUE")

# for item1 in subfolderList:
# 	newSubfolderList = []
# 	for item2 in subfolderList:
# 		if (item2==item1):
# 			testFolder = item2
# 		else:
# 			newSubfolderList.append(item2)
# 	#print newSubfolderList
# 	print "Test Set as: "+ testFolder
# 	sys.stderr.write(testFolder)
	
# 	(combinedFilePaths, combinedLabelList) = getFilesInAllFolders(pathToSpamFolder, newSubfolderList)
# 	#print combinedFilePaths
# 	#pauseKey = raw_input()
# 	vectorizer = TfidfVectorizer(input='filename', stop_words='english', tokenizer=tokenize, binary=True)
# 	X = vectorizer.fit_transform(combinedFilePaths).toarray()

# 	#print "vectorizer fittingTransform done"
# 	clf = LinearDiscriminantAnalysis()
# 	clf.fit(X, np.asarray(combinedLabelList))

# 	(testFilesPathList, testFilesLabelList) = getFilesInFolder(pathToSpamFolder + testFolder)
# 	testX = vectorizer.transform(testFilesPathList).toarray()
# 	#testVectorizer = CountVectorizer(input='filename',vocabulary=vectorizer.vocabulary_)
# 	#testX = testVectorizer.fit_transform(testFilesPathList).toarray()
# 	answer = clf.score(testX, np.asarray(testFilesLabelList))

# 	print answer*100
# 	sys.stderr.write(":" + str(answer*100) + "\n")
# 	print "\n"




# print "NOW FOR ONLY LEMMATIZED, TTFIDF_False, BINARYTRUE"
# sys.stderr.write("NOW FOR ONLY LEMMATIZED, TTFIDF_False, BINARYTRUE")

# for item1 in subfolderList:
# 	newSubfolderList = []
# 	for item2 in subfolderList:
# 		if (item2==item1):
# 			testFolder = item2
# 		else:
# 			newSubfolderList.append(item2)
# 	#print newSubfolderList
# 	print "Test Set as: "+ testFolder
# 	sys.stderr.write(testFolder)
	
# 	(combinedFilePaths, combinedLabelList) = getFilesInAllFolders(pathToSpamFolder, newSubfolderList)
# 	#print combinedFilePaths
# 	#pauseKey = raw_input()
# 	vectorizer = TfidfVectorizer(input='filename', stop_words='english', tokenizer=tokenize, use_idf=False)
# 	X = vectorizer.fit_transform(combinedFilePaths).toarray()

# 	#print "vectorizer fittingTransform done"
# 	clf = LinearDiscriminantAnalysis()
# 	clf.fit(X, np.asarray(combinedLabelList))

# 	(testFilesPathList, testFilesLabelList) = getFilesInFolder(pathToSpamFolder + testFolder)
# 	testX = vectorizer.transform(testFilesPathList).toarray()
# 	#testVectorizer = CountVectorizer(input='filename',vocabulary=vectorizer.vocabulary_)
# 	#testX = testVectorizer.fit_transform(testFilesPathList).toarray()
# 	answer = clf.score(testX, np.asarray(testFilesLabelList))

# 	print answer*100
# 	sys.stderr.write(":" + str(answer*100) + "\n")
# 	print "\n"











# ##TESTING 1
# (combinedFilePaths, combinedLabelList) = getFilesInFolder(pathToSpamFolder+subfolderList[0])
# vectorizer = CountVectorizer(input='filename')
# X = vectorizer.fit_transform(combinedFilePaths).toarray()
# tokens = vectorizer.get_feature_names()

# print "vectorizer fittingTransform done"
# clf = GaussianNB()
# clf.fit(X, np.asarray(combinedLabelList))

# (testFilesPathList, testFilesLabelList) = getFilesInFolder(pathToSpamFolder + "part6/")
# testVectorizer = CountVectorizer(input='filename',vocabulary=vectorizer.vocabulary_)
# testX = testVectorizer.fit_transform(testFilesPathList).toarray()
# answer = clf.score(testX, np.asarray(testFilesLabelList))

# print answer


#TESTING 1
# (combinedFilePaths, combinedLabelList) = getFilesInFolder(pathToSpamFolder+subfolderList[0])
# vectorizer = TfidfVectorizer(input='filename', stop_words='english', tokenizer=tokenize, binary=True)
# X = vectorizer.fit_transform(combinedFilePaths).toarray()
# tokens = vectorizer.get_feature_names()

# print "vectorizer fittingTransform done"
# clf = LinearDiscriminantAnalysis()
# clf.fit(X, np.asarray(combinedLabelList))

# (testFilesPathList, testFilesLabelList) = getFilesInFolder(pathToSpamFolder + "part6/")
# t = vectorizer.transform(testFilesPathList).toarray()
# #testVectorizer = CountVectorizer(input='filename',vocabulary=vectorizer.vocabulary_)
# #testX = testVectorizer.fit_transform(testFilesPathList).toarray()
# answer = clf.score(t, np.asarray(testFilesLabelList))

# print answer








# (spamFiles, nonSpamFiles)=getFilesInFolder(pathToSpamFolder)

# numSpamFiles = len(spamFiles)
# numNonSpamFiles = len(nonSpamFiles)
# print "numSpamFiles: ", numSpamFiles
# print "numNonSpamFiles: ", numNonSpamFiles

# print "Now Vectorizing"
# vectorizer = CountVectorizer(input='filename')
# print "Vectorized"

# vectorizer.fit_transform(spamFiles)
# tokens = vectorizer.get_feature_names()

# print len(tokens)
# #print tokens
