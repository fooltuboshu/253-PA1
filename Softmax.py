import numpy as np 
import matplotlib.pyplot as plt
import re
import random
from dataloader import display_face, images, labels
from sklearn.decomposition import PCA

#get train, validation and test data

#general function for oneHotCoding for a single column vector
def oneHotEncoding(Y):
	oneHotEncodeVariables = {}
	encodedY = []
	distinctVals = set(Y)
	numOfDistintcVals = len(distinctVals)

	for index, value in enumerate(distinctVals):
   		oneHotEncodeVariables[value] = index

	for i in range(len(Y)):
		encodedY.append(np.zeros(numOfDistintcVals))
		encodedY[i][oneHotEncodeVariables[Y[i]]] = 1
	print(oneHotEncodeVariables)

	return encodedY

def getRawMatrix(a): # a is image48
	rowNob = a[0].shape[0]
	colNob = a[0].shape[1]
	dataMatrix = np.zeros((len(a), rowNob*colNob))
	#print(a[0].reshape(rowNob*colNob, 1).shape)
	for it in range(len(a)):
		dataMatrix[it,:] = a[it].reshape(rowNob*colNob, 1).flatten().astype(float)

	return dataMatrix

from sklearn import preprocessing

def applyPCA(d, k):

	data = preprocessing.scale(data)

	dataT = data.T
	# meanDataT = dataT.mean(axis=0).flatten()
	# dataT -= meanDataT
	U, S, VT = np.linalg.svd(dataT, full_matrices=False)

	V = VT.T
	V = V[:,:k]
	S = S[:k]

	PCs = dataT.dot(V)

	for i in range(k):	
		PCs[:, i] = PCs[:, i]/np.sum(PCs[:, i], axis = 0) #np.sqrt(S[i])*
	print(PCs[0])
	pca = PCA(n_components=k, whiten=True)
	pca.fit(data)
	dataNew = pca.transform(data)
	print(np.mean(dataNew, axis=0))
	print(np.std(dataNew, axis=0))
	print("fwe", pca.components_.shape)
	print("PCA singule", pca.singular_values_)

	print("MIND", S)
	# #standardizing the data
	# for i in range(dataT.shape[1]):
	# 	dataT[:, i] = dataT[:, i]/np.sqrt(S[i])


	return dataT.T.dot(PCs)

def cal_cost(theta, X, Y):
	N = X.shape[0]
	c = Y.shape[1]
	htheta = softmax(X.dot(theta))
	cost = -(1/c*N)*np.sum(Y*np.log(htheta))
	return cost

def softmax(X):

	#numerical stability
	output = []
	for i in range(X.shape[0]):
		x = X[i]
		eX = np.exp(x - np.max(x))
		output.append(eX / eX.sum())

	return np.asarray(output)

	# print

def BGD(X, Y, valX, valY):
	N = X.shape[0]
	d = X.shape[1]	
	c = Y.shape[1]
	alpha = 0.01
	num_iteration = 50
	cost = np.zeros((num_iteration, 1))
	costVal = np.zeros((num_iteration, 1))
	theta = np.zeros((d,c))

	for it in range(num_iteration):
		htheta = softmax(X.dot(theta))
		#alpha = 1/(np.sqrt(num_iteration)) * alpha0
		# print("alpha", alpha)
		theta = theta - alpha*(X.T.dot(htheta - Y))
		cost[it] = cal_cost(theta, X, Y)
		costVal[it] = cal_cost(theta, valX, valY)

		if it > 0 and costVal[it] > costVal[it-1]:
			print("early-stopping at ",it)
			break 
		# print(theta)

	return theta, cost, costVal


def splitData(images):
	X = []
	Y = []

	for it in range(len(images)):
		#ignore all happy with teeth and neutral face
		if (re.search(r'ht\d.*',labels[it].split('_')[1]) is None) and \
		(re.search(r'n\d.*',labels[it].split('_')[1]) is  None):
			X.append(images[it])
			Y.append(labels[it])

	
	indices = list(range(len(Y))) #generate the indices
	random.shuffle(indices) #shuffle the indices - equivalent to shuffling the dataset

	validationId = random.choice(indices) #pick one indice randomly for choosing a subject (irrespective of emotion)	
	validationIds = [i for i, x in enumerate(Y) if Y[validationId].split('_')[0] in x] #get indices for all emotions for that subject

	#doing the same for test data
	indices = [x for x in indices if x not in validationIds]
	testId = random.choice(indices)
	testIds = [i for i, x in enumerate(Y) if Y[testId].split('_')[0] in x]

	#finally get the train data
	trainIds = [x for x in indices if x not in testIds]

	#encode the labels
	Y = [x.split('_')[1][0] for x in Y]
	encodedY = Y#oneHotEncoding(Y)

	trainX = getRawMatrix([X[i] for i in trainIds])
	trainY = np.asarray([encodedY[i] for i in trainIds])


	validationX = getRawMatrix([X[i] for i in validationIds])
	validationY = np.asarray([encodedY[i] for i in validationIds])


	testX = getRawMatrix([X[i] for i in testIds])
	testY = np.asarray([encodedY[i] for i in testIds])


	return trainX, trainY, validationX, validationY, testX, testY


def getAccuracy(theta, testX, testY):

	pred =  (softmax(testX.dot(theta)))
	correct = []
	# print(pred[0].argmax(), pred[0])
	# print(testY[0].argmax(), testY[0])
	
	for i in range(len(pred)):

		if pred[i].argmax()==testY[i].argmax():
			correct.append(1)
		else:
			correct.append(0)
	print(len(correct))

	return sum(correct)/len(correct)

accuracies = []


############### DEBUGGING PURPOSE ONLY #####################

from sklearn.linear_model import LogisticRegression

acc = []

for i in range(10):
	trainX, trainY, validationX, validationY, testX, testY = splitData(images)

	mean = np.mean(trainX, axis = 0 )
	std = np.std(trainX, axis = 0)
	scaler = preprocessing.StandardScaler().fit(trainX)

	trainX = scaler.transform(trainX)

	pca = PCA(n_components=30, whiten=True)
	pca.fit(trainX)
	trainXPCA = pca.transform(trainX)
	trainXPCA =  np.c_[np.ones((trainXPCA.shape[0],1)),trainXPCA]


	#validationX = (validationX-mean)/std
	validationXPCA = pca.transform(scaler.transform(validationX))
	validationXPCA =  np.c_[np.ones((validationXPCA.shape[0],1)),validationXPCA]

	#testX = (testX-mean)/std
	testXPCA = pca.transform(scaler.transform(testX))
	testXPCA =  np.c_[np.ones((testXPCA.shape[0],1)),testXPCA]

	clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(trainXPCA, trainY)
	acc.append(clf.score(testXPCA, testY))

print(np.mean(acc))
print(np.std(acc))

# for i in range(10):
# 	trainX, trainY, validationX, validationY, testX, testY = splitData(images)

# 	# preprocess()
# 	mean = np.mean(trainX, axis = 0 )
# 	std = np.std(trainX, axis = 0)
# 	trainX = preprocessing.scale(trainX)

# 	pca = PCA(n_components=30, whiten=True)
# 	pca.fit(trainX)
# 	trainXPCA = pca.transform(trainX)
# 	trainXPCA =  np.c_[np.ones((trainXPCA.shape[0],1)),trainXPCA]


# 	validationX = (validationX-mean)/std
# 	validationXPCA = pca.transform(validationX)
# 	validationXPCA =  np.c_[np.ones((validationXPCA.shape[0],1)),validationXPCA]

# 	testX = (testX-mean)/std
# 	testXPCA = pca.transform(testX)
# 	testXPCA =  np.c_[np.ones((testXPCA.shape[0],1)),testXPCA]

# 	theta, cost, costVal = BGD(trainXPCA, trainY, validationXPCA, validationY)
	
# 	accuracies.append(getAccuracy(theta, testXPCA, testY))

# print(np.mean(accuracies))
# print(np.std(accuracies))

# # softmax regression
# w = np.zeros([trainX.shape[1], trainY.shape[1]])
# lamda = 1
# iterations = 2
# learningRate = 1e-5
# losses = []
# for i in range(0,iterations):
#     loss,grad = getLoss(w,trainX,trainY,lamda)
#     losses.append(loss)
#     w = w - (learningRate * grad)
# print(loss)


#display_face(reconMatrix[23].reshape(380,240))

#for i in range(6):

# x = dataT.dot(V[0]) 

# #x = 255*(x-min(x))/(max(x)-min(x))
# x =  np.array(x.reshape(380,240))
# plt.imshow(x, cmap='gray')
# plt.show()


# X = 2*np.random.rand(100, 1)
# Y = 3 + 10*X + 0.5*np.random.randn(100, 1)



# # print(X.shape[0])
# X_b = np.c_[np.ones((len(X), 1)), X]



# theta, cost_his = BGD(X_b, Y)
# print(theta)


# plt.figure(1)
# x_fit = np.arange(0.0, 2.0, 0.02)
# plt.plot(X ,Y, 'bo', x_fit, theta[0] + theta[1]*x_fit, 'k') # 'bo' 'k'
# plt.title('original data')
# plt.ylabel('Y')
# plt.xlabel('X')
# plt.text(0.2, 20, txt)# , ha='left'
# plt.show()

# print(Y[2,:])
# def SGD(X_b, Y):
# 	m = X_b.shape[0]
# 	alpha = 0.01
# 	num_iteration = 1500
# 	cost = np.zeros((num_iteration, 1))
# 	theta = np.random.randn(2,1)

# 	for it in range(num_iteration):
# 		cost_temp = 0.0
# 		for i in range(m):
# 			hthetai = X_b[i, :].dot(theta)
# 			#print(hthetai)
# 			#print(X_b.T[:, i].T)
# 			theta = theta - (1/m)*alpha*(X_b[i,:].reshape(2,1)*(hthetai - Y[i]))

# 			#print(theta)
# 			cost_temp += cal_cost(theta, X_b[i, :], Y[i, :])
# 		cost[it] = cost_temp

# 	return theta, cost

# thetaS, _, = SGD(X_b, Y)
# print(thetaS)




