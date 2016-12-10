import os, sys, string, random, shutil
import numpy as np
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# from nltk.corpus import stopwords
# from nltk import download
# download('stopwords')

questionPart = 'A' # Set as A, B, C for the part of the question
# questionPart = 'B'
# questionPart = 'C'
# classifier = 'Gaussian'
# classifier = 'Multinomial'
# classifier = 'SVC'
classifier = 'LinearSVC'
numCrossValidation = 5



pathToSpamFolder = "/home/deepak/Acads/Sem8/MLT/Assignments/Assgn2/Data/bare/"
subfolderList = ['part1/', 'part2/','part3/','part4/','part5/','part6/','part7/','part8/','part9/','part10/']
splitFolders = ['1/', '2/', '3/', '4/', '5/']
stemmer = PorterStemmer()


#TESTING FOR MULTINOMIAL + GAUSSIAN NAIVE BAYES
if(classifier == 'Gaussian'):
	from sklearn.naive_bayes import GaussianNB			# Gaussian Naive Bayes
elif (classifier == 'Multinomial'):
	from sklearn.naive_bayes import MultinomialNB		# Multinomial Naive Bayes
elif (classifier == 'SVC'):
	from sklearn.svm import SVC
elif (classifier == 'LinearSVC'):
	from sklearn.svm import LinearSVC
else:
	print "Give Proper Classifier"


def restructureDirectory():
	global subfolderList
	random.shuffle(subfolderList)
	global splitFolders
	t=0
	for i in splitFolders:
		fileList = []
		if os.path.exists(pathToSpamFolder+i):
			shutil.rmtree(pathToSpamFolder+i)
		os.makedirs(pathToSpamFolder+i)
		fileList = fileList + getFilesInFolder(pathToSpamFolder+subfolderList[2*t])[0]
		fileList = fileList + getFilesInFolder(pathToSpamFolder+subfolderList[2*t+1])[0]
		print subfolderList[2*t], subfolderList[2*t+1]
		for filePath in fileList:
			shutil.copyfile(filePath, pathToSpamFolder+i+os.path.basename(filePath))
		t=t+1

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


# restructureDirectory()

for item1 in splitFolders:
	newSubfolderList = []
	for item2 in splitFolders:
		if (item2==item1):
			testFolder = item2
		else:
			newSubfolderList.append(item2)
	#print newSubfolderList
	print "Test Set as: "+ testFolder
	# sys.stderr.write(testFolder)
	
	(combinedFilePaths, combinedLabelList) = getFilesInAllFolders(pathToSpamFolder, newSubfolderList)
	# print len(combinedFilePaths)

	if(questionPart == 'A'):
		vectorizer = CountVectorizer(input='filename')    # PART A
	elif (questionPart == 'B'):
		vectorizer = CountVectorizer(input='filename', stop_words='english')   # PART B
	elif (questionPart == 'C'):
		vectorizer = CountVectorizer(input='filename', stop_words='english', tokenizer=tokenize)  # PART C
	else:
		print "ERROR IN QUESTION PART"
	
	X = vectorizer.fit_transform(combinedFilePaths).toarray()
	(testFilesPathList, testFilesLabelList) = getFilesInFolder(pathToSpamFolder + testFolder)
	testX = vectorizer.transform(testFilesPathList).toarray()

	if (classifier == 'Gaussian'):
		clf = GaussianNB()
	elif (classifier == 'Multinomial'):
		clf = MultinomialNB()
	elif (classifier == 'SVC'):
		clf = SVC()
	elif (classifier == 'LinearSVC'):

		# This region of code was used for tuning the parameter C

		# for C in [0.01, 0.1, 0.5, 1, 10, 100, 1000, 10000, 100000, 1000000, 100000000]:
		# 	clf = LinearSVC(loss='hinge', C=C)
		# 	# clf2 = LinearSVC(C=10000000)

		# 	clf.fit(X, np.asarray(combinedLabelList))
		# 	# clf2.fit(X, np.asarray(combinedLabelList))

		# 	answer = clf.score(testX, np.asarray(testFilesLabelList))
		# 	# answer2 = clf2.score(testX, np.asarray(testFilesLabelList))
		# 	print "C=", C, ", Score Hinge: ", answer*100
		# 	# print "Score Standard Formulation", answer2*100

		clf = LinearSVC(loss='hinge', C=0.01)
		clf2 = LinearSVC(loss='hinge', C=10000000)

		clf.fit(X, np.asarray(combinedLabelList))
		clf2.fit(X, np.asarray(combinedLabelList))

		answer = clf.score(testX, np.asarray(testFilesLabelList))
		answer2 = clf2.score(testX, np.asarray(testFilesLabelList))
		print "Score Hinge: ", answer*100
		print "Score Standard Formulation", answer2*100
	else:
		print "Error Clf Not Specified"
	# sys.stderr.write(":" + str(answer*100) + "\n")








# #TESTING FOR GAUSSIAN NAIVE BAYES
# from sklearn.naive_bayes import GaussianNB
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

# 	vectorizer = CountVectorizer(input='filename', stop_words='english', tokenizer=tokenize)
# 	X = vectorizer.fit_transform(combinedFilePaths).toarray()

# 	clf = GaussianNB()
# 	clf.fit(X, np.asarray(combinedLabelList))

# 	(testFilesPathList, testFilesLabelList) = getFilesInFolder(pathToSpamFolder + testFolder)
# 	testX = vectorizer.transform(testFilesPathList).toarray()
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


#TESTING 2
# (combinedFilePaths, combinedLabelList) = getFilesInFolder(pathToSpamFolder+subfolderList[0])
# vectorizer = CountVectorizer(input='filename', stop_words='english', tokenizer=tokenize )
# X = vectorizer.fit_transform(combinedFilePaths).toarray()
# tokens = vectorizer.get_feature_names()

# print "vectorizer fittingTransform done"
# clf = GaussianNB()
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