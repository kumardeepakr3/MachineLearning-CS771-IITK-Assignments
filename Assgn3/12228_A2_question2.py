import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from collections import Counter

dataFileName = "/home/deepak/Acads/Sem8/MLT/Assignments/Assgn3/connect-4.data"

dividedLearningSet = []
numCrossValidation = 5


def getLearningSet():
	with open(dataFileName) as dataFile:
		learningSetList = []
		for line in dataFile:
			csvList = line.strip().split(',')
			# print csvList[42]
			if(csvList[42] == 'win'):
				csvList[42] = 1
			elif(csvList[42] == 'draw'):
				csvList[42] = 0
			else:	#LOSS
				csvList[42] = -1
			learningSetList.append(csvList)
	return learningSetList


def updateLearningSet(learningSetList):
	featureMatrix = []
	labelList = []
	for vector in learningSetList:
		newVector = [0]*(42*3)	# [0,0,0,..0] 43 times
		for i in range(len(vector)):
			feature = vector[i]
			if feature == 'o':
				newVector[3*(i+1)-1] = 1
			elif feature == 'b':
				newVector[3*(i+1)-1-1] = 1
			elif feature == 'x':
				newVector[3*(i+1)-2-1] = 1
			else:	# For the label
				labelList.append(feature)
		featureMatrix.append(newVector)
	return featureMatrix,labelList


# Taken from http://stackoverflow.com/questions/2130016/splitting-a-list-of-arbitrary-size-into-only-roughly-n-equal-parts
def chunkIt(seqX, seqY, num):
  avg = len(seqX) / float(num)
  outX = []
  outY = []
  last = 0.0
  while last < len(seqX):
    outX.append(seqX[int(last):int(last + avg)])
    outY.append(seqY[int(last):int(last + avg)])
    last += avg
  return zip(outX, outY)


def divideLearningSet(X,y):
	global dividedLearningSet
	dividedLearningSet = chunkIt(X,y,numCrossValidation)


def runOVO():
	print "Running One vs One"
	for i in range(numCrossValidation):
		testLabels = []
		testVectors = []
		trainingLabels = []
		trainingVectors = []
		print "Part", i, " as test set"
		for j in range(len(dividedLearningSet)):
			if(j == i):
				testVectors = dividedLearningSet[j][0]
				testLabels = dividedLearningSet[j][1]
			else:
				trainingVectors += dividedLearningSet[j][0]
				trainingLabels += dividedLearningSet[j][1]
		print "Num of Training Vectors: ", len(trainingVectors), "|    Num of Test Vectors: ",len(testVectors)

		combinedTrainData = zip(trainingVectors, trainingLabels)
		
		winDrawTrainVectors = []
		drawLoseTrainVectors = []
		loseWinTrainVectors = []
		winDrawTrainLabels = []
		drawLoseTrainLabels = []
		loseWinTrainLabels = []

		for x in combinedTrainData:
			if (x[1] == 1 or x[1] ==0): #Win or Draw
				winDrawTrainVectors.append(x[0])
				winDrawTrainLabels.append(x[1])
			if (x[1] == 0 or x[1] ==-1): #Draw or Lose
				drawLoseTrainVectors.append(x[0])
				drawLoseTrainLabels.append(x[1])
			if (x[1] == -1 or x[1] ==1): #Lose or Win
				loseWinTrainVectors.append(x[0])
				loseWinTrainLabels.append(x[1])

		# print len(winDrawTrainVectors), len(drawLoseTrainVectors), len(loseWinTrainVectors)

		clf1 = LinearSVC()
		clf2 = LinearSVC()
		clf3 = LinearSVC()

		#Class 'win' vs 'Draw'
		clf1.fit(winDrawTrainVectors, winDrawTrainLabels)

		#Class 'draw' vs 'Lose'
		clf2.fit(drawLoseTrainVectors, drawLoseTrainLabels)

		#Class 'loss' vs 'win'
		clf3.fit(loseWinTrainVectors, loseWinTrainLabels)

		winDrawPredict = clf1.predict(testVectors)
		winDrawDistMargin = clf1.decision_function(testVectors)
		drawLosePredict = clf2.predict(testVectors)
		drawLoseDistMargin = clf2.decision_function(testVectors)
		loseWinPredict = clf3.predict(testVectors)
		loseWinDistMargin = clf3.decision_function(testVectors)

		combinedDistMargin = zip(winDrawDistMargin, drawLoseDistMargin, loseWinDistMargin)
		combinedPredictions = zip(winDrawPredict, drawLosePredict, loseWinPredict)
		ovoPredictedLabels = []
		# countdict = {1:0, 2:0, 3:0}
		for element in combinedPredictions:
			ovoPredictedLabels.append(Counter(element).most_common(1)[0][0])
			# countdict[Counter(element).most_common(1)[0][1]] +=1
		# print countdict

		correctPredictions = 0
		for k in range(len(ovoPredictedLabels)):
			if(ovoPredictedLabels[k] == testLabels[k]):
				correctPredictions +=1
		print correctPredictions/float(len(ovoPredictedLabels))
		print


def runOVR():
	print "Running One vs Rest"
	for i in range(numCrossValidation):
		testLabels = []
		testVectors = []
		trainingLabels = []
		trainingVectors = []
		print "Part", i, " as test set"
		for j in range(len(dividedLearningSet)):
			if(j == i):
				testVectors = dividedLearningSet[j][0]
				testLabels = dividedLearningSet[j][1]
			else:
				trainingVectors += dividedLearningSet[j][0]
				trainingLabels += dividedLearningSet[j][1]
		print "Num of Training Vectors: ", len(trainingVectors), "|    Num of Test Vectors: ",len(testVectors)
		clf1 = LinearSVC()
		clf2 = LinearSVC()
		clf3 = LinearSVC()
		#Class 'win' vs Rest
		winTrainLabels = [1 if x==1 else 0 for x in trainingLabels]
		clf1.fit(trainingVectors, winTrainLabels)

		#Class 'draw' vs Rest
		drawTrainLabels = [1 if x==0 else 0 for x in trainingLabels]
		clf2.fit(trainingVectors, drawTrainLabels)

		#Class 'loss' vs Rest
		lossTrainLabels = [1 if x==-1 else 0 for x in trainingLabels]
		clf3.fit(trainingVectors, lossTrainLabels)

		winPredict = clf1.predict(testVectors)
		winDistMargin = clf1.decision_function(testVectors)
		drawPredict = clf2.predict(testVectors)
		drawDistMargin = clf2.decision_function(testVectors)
		losePredict = clf3.predict(testVectors)
		loseDistMargin = clf3.decision_function(testVectors)

		combinedDistMargin = zip(winDistMargin, drawDistMargin, loseDistMargin)
		ovrPredictedLabels = []
		for element in combinedDistMargin:
			ovrPredictedLabels.append(1-element.index(max(element)))

		correctPredictions = 0
		for k in range(len(ovrPredictedLabels)):
			if(ovrPredictedLabels[k] == testLabels[k]):
				correctPredictions +=1
		print correctPredictions/float(len(ovrPredictedLabels))
		print


def runStandardSVM():
	for i in range(numCrossValidation):
		testLabels = []
		testVectors = []
		trainingLabels = []
		trainingVectors = []
		print "Part", i, " as test set"
		for j in range(len(dividedLearningSet)):
			if(j == i):
				testVectors = dividedLearningSet[j][0]
				testLabels = dividedLearningSet[j][1]
			else:
				trainingVectors += dividedLearningSet[j][0]
				trainingLabels += dividedLearningSet[j][1]
		print "Num of Training Vectors: ", len(trainingVectors), "|    Num of Test Vectors: ",len(testVectors)
		clf = LinearSVC() # Default is OVR
		clf2 = SVC(kernel='linear')
		clf.fit(trainingVectors, trainingLabels)
		clf2.fit(trainingVectors, trainingLabels)
		score1 = clf.score(testVectors, testLabels)
		score2 = clf2.score(testVectors, testLabels)
		print "OVR Standard Score: ", score1
		print "OVO Standard Score: ", score2


learningSetList = getLearningSet()
all_X,all_y = updateLearningSet(learningSetList)
divideLearningSet(all_X,all_y)
print "Num of Parts(split for cross-validation): ", len(dividedLearningSet)
print "Length of each divided part: ", len(dividedLearningSet[0][1]), len(dividedLearningSet[1][1]), len(dividedLearningSet[2][1]), len(dividedLearningSet[3][1]), len(dividedLearningSet[4][1])
print "Total Number of Vectors in Learning Set: ", len(all_X), "\n"

runOVR()
runOVO()
runStandardSVM()

# for i in range(numCrossValidation):
# 	testLabels = []
# 	testVectors = []
# 	trainingLabels = []
# 	trainingVectors = []
# 	print "Part", i, " as test set"
# 	for j in range(len(dividedLearningSet)):
# 		if(j == i):
# 			testVectors = dividedLearningSet[j][0]
# 			testLabels = dividedLearningSet[j][1]
# 		else:
# 			trainingVectors += dividedLearningSet[j][0]
# 			trainingLabels += dividedLearningSet[j][1]
# 	print len(trainingVectors), len(trainingLabels)
# 	print len(testVectors), len(testLabels)
# 	clf = LinearSVC()
# 	clf.fit(trainingVectors, trainingLabels)
# 	score = clf.score(testVectors, testLabels)
# 	print score

