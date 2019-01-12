from dataloader import display_face, images, labels, happyIndices, \
sadIndices, happytthIndices, surpriseIndices, \
fearIndices, angerIndices, disgustIndices, neutralIndices
import numpy as np
import re
# display_face(images[happytthIndices[3]])

# print((images[0].shape))
# print(happyIndices)
# print(sadIndices)
# display_face(images[0])
# display_face(images[1])
# print(labels)
pepList = ['018', '027', '036', '037', '041', '043', '044', '048', '049', '050']

# print(type(images))

def getImage48(images):
	newImage48 = []
	randPepList = []
	randPepIndices = np.random.permutation(10)[:8] #get the random 8 people form 10
	for it in range(8):
		randPepList.append(pepList[randPepIndices[it]])
	for it in range(80):
		if labels[it].split('_')[0] in randPepList and \
		(re.search(r'ht\d.*',labels[it].split('_')[1]) is None) and \
		(re.search(r'n\d.*',labels[it].split('_')[1]) is  None):
			newImage48.append(images[it])
	return newImage48, randPepList

newImage48, randPepList = getImage48(images)
print(randPepList)
print(len(newImage48))
# display_face(newImage48[0])

# def getLastTwo(randPepList, pepList):
# 	lastTwoPep = []
# 	for it in range(len(pepList)):
# 		if pepList[it] is not in randPepList:
# 			lastTwoPep.append(pepList[it])
# 	return lastTwoPep
# lastTwoPep = getLastTwo(randPepList, pepList)
lastTwoPep = list(set(pepList) - set(randPepList)) # get the last two people
print(lastTwoPep)

print(newImage48[0].shape)

def getRawMatrix(a): # a is image48
	rowNob = a[0].shape[0]
	colNob = a[0].shape[1]
	dataMatrix = np.zeros((42, rowNob*colNob))
	#print(a[0].reshape(rowNob*colNob, 1).shape)
	for it in range(42):
		dataMatrix[it,:] = a[it].reshape(rowNob*colNob, 1).flatten()

	return dataMatrix
print(getRawMatrix(newImage48).shape)
data = getRawMatrix(newImage48)
# print(data - data.mean(axis = 0))

def PCA(dataMatrix):
	dataMatrix -= dataMatrix.mean(axis = 0)#.ravel() #.flatten()
	mag_data = np.dot(dataMatrix, dataMatrix.T)
	evals, evecs = np.linalg.eig(mag_data)

	idx = np.argsort(evals)[::-1]

	evecs = evecs[:,idx]/np.linalg.norm(evecs[:,idx])#
	evals = evals[idx]	
	newEvecs = np.dot(dataMatrix.T, evecs)

	return newEvecs
eves = PCA(data)
print(eves[:,1])
def displayEigFace(evecs):
	K = 3
	EigFace = []
	for it in range(42):
		EigFace.append(evecs[:, it].reshape(380, 240))
	print(EigFace[1].shape)
	for it in range(K):
		display_face(EigFace[it])

displayEigFace(PCA(getRawMatrix(newImage48)))




