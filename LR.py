from dataloader import display_face, images, labels, happyIndices, \
sadIndices, happytthIndices, surpriseIndices, \
fearIndices, angerIndices, disgustIndices, neutralIndices

from PCA import randPepList, lastTwoPep, pepList, getRawMatrix, \
PCA_MH, plot_eignFace

import numpy as np

# def getTrainIndices():
# 	trainHappyIndices = [];
# 	trainSadIndices = []
# 	for it in range(10):
# 		if labels[happyIndices[it]].split('_')[0] in randPepList:
# 			trainHappyIndices.append(happyIndices[it])
# 		if labels[sadIndices[it]].split('_')[0] in randPepList:
# 			trainSadIndices.append(sadIndices[it])
# 	return trainHappyIndices, trainSadIndices
# tHI, tSI = getTrainIndices();

# print(pepList)
# print(happyIndices)
# print(tHI)

def getHoldOutSet():
	temp_index = np.random.randint(10)
	print(pepList[temp_index])
	indices = [i for i, s in enumerate(labels) if \
	pepList[temp_index] in s]
	print(indices)
	for i in range(8):
		if indices[i] in happyIndices:
			happyHoldSet = indices[i]
			happyIndices.remove(indices[i])
		if indices[i] in sadIndices:
			sadIndices.remove(indices[i])
			sadHoldSet = indices[i]

	# holdOutPep = pepList[temp_index] #
	# happyIndices.pop(temp_index)
	# sadIndices.pop(temp_index)
	return happyHoldSet, sadHoldSet

# print(getHoldOutSet())

hHS, sHS = getHoldOutSet()

def getTrainSet():
	tSet = []
	tImage = []
	teImage = []
	temp_List = happyIndices + sadIndices
	temp_index = np.random.permutation(18)[:16]
	for i in range(16):
		tSet.append(temp_List[temp_index[i]])
		tImage.append(images[temp_List[temp_index[i]]])
	# trainSet = temp_List[temp_index]
	teSet = list(set(temp_List) - set(tSet))
	for i in range(2):
		teImage.append(images[teSet[i]]) 
	return tSet, tImage, teSet, teImage
	# 
	# trainHIndices = happyIndices[:, temp_index]
	# temp_index = np.random.permutation(9)[:8]
	# trainSIndices = sadIndices[:, temp_index]
trainSet, trainImages, testSet, testImages = getTrainSet() 		# get training Set Indices
print(labels)
print(trainSet)
print(len(trainImages))
data16Images = getRawMatrix(trainImages)

def getPCAweight(data_images, k):
	eMatrix = PCA_MH(data_images, k)
	print(eMatrix.shape)

getPCAweight(data16Images, 10)




