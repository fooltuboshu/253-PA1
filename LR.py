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
# print(labels)
# print(trainSet)
# print(len(trainImages))
print('testset', testSet[0])
data16Images = getRawMatrix(trainImages)

def getPCAweight(data_images, k):
	eMatrix = PCA_MH(data_images, k)
	# print(eMatrix.shape)
	weight_temp = data_images.dot(eMatrix)
	return weight_temp

weight_16 = getPCAweight(data16Images, 10)
print('weight_16', weight_16)

def getRawY(set):
	y = np.zeros(len(set))
	for i in range(len(set)):
		if set[i] in sadIndices:
			y[i] = 1
	return y
 
def one_hot(Y, nom_samples, nom_features):
	one_hot = np.zeros((nom_samples, nom_features))
	for i in range(nom_samples):
		one_hot[i, int(y[i])] = 1
	return one_hot

y = getRawY(trainSet)
# print(y)
# print(one_hot(y, 16, 2))
	
def getW_b(weight):
	W_b = np.c_[np.ones((len(weight), 1)), weight]
	return W_b

def cal_cost(theta, X_b, Y):
	m = X_b.shape[0]
	cost = 0
	htheta = 1/(1+np.exp(-X_b.dot(theta)))
	# print('htheat', htheta)
	# for i in range(m):
	# 	if Y[i, 0] == 1:
	# 		cost += np.log(htheta[i])
	# 	else:
	# 		cost += np.log(1 - htheta[i])

	cost = - (1/m)*np.sum(y*np.log(htheta.T))
	return cost

def LR(X, Y):
	X_b = np.c_[np.ones((len(X), 1)), X]
	N = X_b.shape[0]
	d = X_b.shape[1]	
	alpha0 = 1
	num_iteration = 10
	cost = np.zeros((num_iteration, 1))
	theta = np.random.randn(d,1)

	for it in range(num_iteration):
		htheta = 1/(1+np.exp(-X_b.dot(theta)))
		alpha = 1/(np.sqrt(num_iteration)) * alpha0
		theta = theta - alpha*(X_b.T.dot(htheta - Y))
		cost[it] = cal_cost(theta, X_b, Y)
	weight = theta[1:d]
	bias = theta[0]
	return cost, weight, bias

def compute_scores(x, weight, bias):
	return x*weight + bias

def predict_HorS(x):
	y = 1/(1 + np.exp(-x))
	if y > 0.5:
		print('Its a happy face')
	else:
		print('Its a sad face')

def main():
	Y = one_hot(y, 16, 2)
	_, newW, newBias = LR(weight_16, Y)
	print(testImages[0].shape)
	print(type(testImages))
	# print(len(images))
	dataTestImage = getRawMatrix(testImages[0])
	weightTestImage = getPCAweight(dataTestImage)
	scoresTest = compute_scores(weightTestImage, newW, newBias)
	predict_HorS(scoresTest)

main() 


