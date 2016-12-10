from sklearn.neighbors import KNeighborsClassifier
import os, sys, string, struct
import numpy as np
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
# from nltk import download
# download('punkt')



pathToMNISTFolder = "/home/deepak/Acads/Sem8/MLT/Assignments/Assgn2/Data/"
pathToSpamFolder = "/home/deepak/Acads/Sem8/MLT/Assignments/Assgn2/Data/bare/"
subfolderList = ['part1/', 'part2/','part3/','part4/','part5/','part6/','part7/','part8/','part9/','part10/']
# stemmer = PorterStemmer()


#I have taken this snippet of code directly from from the following blog
# Taken from http://g.sweyla.com/blog/2012/mnist-numpy/
def load_mnist(dataset="training", digits=np.arange(10), path="."):
	"""
	Loads MNIST files into 3D numpy arrays
	Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
	"""
	if dataset == "training":
		fname_img = os.path.join(path, 'train-images.idx3-ubyte')
		fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
	elif dataset == "testing":
		fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
		fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
	else:
		raise ValueError("dataset must be 'testing' or 'training'")

	flbl = open(fname_lbl, 'rb')
	magic_nr, size = struct.unpack(">II", flbl.read(8))
	lbl = pyarray("b", flbl.read())
	flbl.close()

	fimg = open(fname_img, 'rb')
	magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
	img = pyarray("B", fimg.read())
	fimg.close()

	ind = [ k for k in range(size) if lbl[k] in digits ]
	N = len(ind)

	images = zeros((N, rows, cols), dtype=uint8)
	labels = zeros((N, 1), dtype=int8)
	for i in range(len(ind)):
		images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
		labels[i] = lbl[ind[i]]

	return images, labels





train_images, train_labels = load_mnist(dataset="training", path=pathToMNISTFolder)
test_images, test_labels = load_mnist(dataset="testing", path=pathToMNISTFolder)
n_samples = len(train_images)
X = train_images.reshape((n_samples, -1))
y = train_labels.reshape(len(train_labels),)
n_test_samples = len(test_images)
testX = test_images.reshape((n_test_samples, -1))
testY = test_labels.reshape(len(test_labels))
print "Model Load Done"


kListRun = [1,2,3,4]
metricList = ['hamming', 'chebyshev', 'manhattan', 'euclidean', 'minkowski']

for currMetric in metricList:
	for k in kListRun:
		print currMetric
		neigh = KNeighborsClassifier(n_neighbors=k, metric=currMetric)
		neigh.fit(X, y)
		print "Model Fitting Done"
		scores = neigh.score(testX, testY)
		print str(currMetric) + " K=" + str(k) + " Score:" +str(scores)







# # Run for Metric Euclidean
# for k in kListRun:
# 	print "Euclidean"
# 	neigh = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
# 	neigh.fit(X, y)
# 	print "Model Fitting Done"
# 	scores = neigh.score(testX, testY)
# 	print k, scores
# 	print

# # print train_labels.asarray()

# print

# # Run for Metric manhattan
# for k in kListRun:
# 	print "Manhattan"
# 	neigh = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
# 	neigh.fit(X, y)
# 	print "Model Fitting Done"
# 	scores = neigh.score(testX, testY)
# 	print k, scores
# 	print

# print

# # Run for Metric manhattan
# for k in kListRun:
# 	print "ChebyShev"
# 	neigh = KNeighborsClassifier(n_neighbors=k, metric='chebyshev')
# 	neigh.fit(X, y)
# 	print "Model Fitting Done"
# 	scores = neigh.score(testX, testY)
# 	print k, scores
# 	print





# #Taken from http://stackoverflow.com/questions/26126442/combining-text-stemming-and-removal-of-punctuation-in-nltk-and-scikit-learn
# def stem_tokens(tokens, stemmer):
# 	stemmed = []
# 	for item in tokens:
# 		stemmed.append(stemmer.stem(item))
# 	return stemmed


# #Taken from http://stackoverflow.com/questions/26126442/combining-text-stemming-and-removal-of-punctuation-in-nltk-and-scikit-learn
# def tokenize(text):
# 	text = "".join([ch for ch in text if ch not in string.punctuation])
# 	tokens = word_tokenize(text)
# 	stems = stem_tokens(tokens, stemmer)
# 	return stems


# def getFilesInFolder(pathOfFolder):
# 	allFileFullPathList = []
# 	labelList = []
# 	for file in os.listdir(pathOfFolder):
# 		allFileFullPathList.append(pathOfFolder+file)
# 		#print file
# 		if file.startswith("spm"):
# 			labelList.append(1)  # Spam
# 		else:
# 			labelList.append(0)  # NonSpam
# 	#print nonSpamFiles
# 	#print "\n\n\n"
# 	#print spamFiles
# 	return (allFileFullPathList, labelList)


# def getFilesInAllFolders(pathOfParent, listOfChildNames):
# 	combinedFilePaths = []
# 	combinedLabelList = []
# 	for childFolder in listOfChildNames:
# 		(allFileFullPathList, labelList) = getFilesInFolder(pathOfParent+childFolder)
# 		combinedFilePaths += allFileFullPathList
# 		combinedLabelList += labelList
# 	return (combinedFilePaths, combinedLabelList)






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