from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog
import os, sys, string, struct, pickle
import numpy as np
from array import array


pathToMNISTFolder = "/home/deepak/Acads/Sem8/MLT/Assignments/Assgn2/Data/"


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
	lbl = array("b", flbl.read())
	flbl.close()

	fimg = open(fname_img, 'rb')
	magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
	img = array("B", fimg.read())
	fimg.close()

	ind = [ k for k in range(size) if lbl[k] in digits ]
	N = len(ind)
	images = np.zeros((N, rows, cols), dtype=np.uint8)
	labels = np.zeros((N, 1), dtype=np.int8)
	for i in range(len(ind)):
		images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
		labels[i] = lbl[ind[i]]

	return images, labels



def generateHogOfAllImages(imagesArray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(3,3)):
	hogVectorsList = []
	for image in imagesArray:
		hogVector = hog(image, orientations, pixels_per_cell, cells_per_block)
		hogVectorsList.append(hogVector)
	return np.asarray(hogVectorsList)


if __name__ == "__main__":
	train_images, train_labels = load_mnist(dataset="training", path=pathToMNISTFolder)
	test_images, test_labels = load_mnist(dataset="testing", path=pathToMNISTFolder)
	n_train_samples = len(train_images)
	n_test_samples = len(test_images)
	print n_train_samples, n_test_samples

	X = generateHogOfAllImages(train_images)
	Y = train_labels.reshape(len(train_labels),)
	testX = generateHogOfAllImages(test_images)
	testY = test_labels.reshape(len(test_labels))
	print "Data Loaded & Hog Found"
	print X.shape, testX.shape
	NumTreeList = [50, 100, 150, 200, 250, 275, 300, 325, 350, 400]
	for numTree in NumTreeList:
		clf = RandomForestClassifier(numTree)
		clf.fit(X, Y)
		score = clf.score(testX, testY)
		print numTree, " :", score*100



# kListRun = [1,2,3,4]
# metricList = ['hamming', 'chebyshev', 'manhattan', 'euclidean', 'minkowski']

# for currMetric in metricList:
# 	for k in kListRun:
# 		print currMetric
# 		neigh = KNeighborsClassifier(n_neighbors=k, metric=currMetric)
# 		neigh.fit(X, y)
# 		print "Model Fitting Done"
# 		scores = neigh.score(testX, testY)
# 		print str(currMetric) + " K=" + str(k) + " Score:" +str(scores)





