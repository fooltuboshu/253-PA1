import numpy as np 
import matplotlib.pyplot as plt

txt = "This is some figure caption"

X = 2*np.random.rand(100, 1)
Y = 3 + 10*X + 0.5*np.random.randn(100, 1)



# print(X.shape[0])
X_b = np.c_[np.ones((len(X), 1)), X]

def cal_cost(theta, X_b, Y):
	m = X_b.shape[0]
	htheta = X_b.dot(theta)
	cost = (1/2*m)*np.sum(np.square(htheta - Y))
	return cost

def BGD(X_b, Y):
	N = X_b.shape[0]
	d = X_b.shape[1]	
	alpha0 = 1
	num_iteration = 100
	cost = np.zeros((num_iteration, 1))
	theta = np.random.randn(d,1)/10

	for it in range(num_iteration):
		htheta = X_b.dot(theta)
		alpha = 1/(np.sqrt(num_iteration)) * alpha0
		theta = theta - (1/N)*alpha*(X_b.T.dot(htheta - Y))
		cost[it] = cal_cost(theta, X_b, Y)

	return theta, cost

theta, cost_his = BGD(X_b, Y)
print(theta)


plt.figure(1)
x_fit = np.arange(0.0, 2.0, 0.02)
plt.plot(X ,Y, 'bo', x_fit, theta[0] + theta[1]*x_fit, 'k') # 'bo' 'k'
plt.title('original data')
plt.ylabel('Y')
plt.xlabel('X')
plt.text(0.2, 20, txt)# , ha='left'
plt.show()

print(Y[2,:])
def SGD(X_b, Y):
	m = X_b.shape[0]
	alpha = 0.01
	num_iteration = 1500
	cost = np.zeros((num_iteration, 1))
	theta = np.random.randn(2,1)

	for it in range(num_iteration):
		cost_temp = 0.0
		for i in range(m):
			hthetai = X_b[i, :].dot(theta)
			#print(hthetai)
			#print(X_b.T[:, i].T)
			theta = theta - (1/m)*alpha*(X_b[i,:].reshape(2,1)*(hthetai - Y[i]))

			#print(theta)
			cost_temp += cal_cost(theta, X_b[i, :], Y[i, :])
		cost[it] = cost_temp

	return theta, cost

thetaS, _, = SGD(X_b, Y)
print(thetaS)

# plt.figure(2)
# x_fit = np.arange(0.0, 2.0, 0.02)
# plt.plot(X ,Y, 'bo', x_fit, thetaS[0] + thetaS[1]*x_fit, 'k') # 'bo' 'k'
# plt.title('original data')
# plt.ylabel('Y')
# plt.xlabel('X')
# plt.text(0.2, 20, txt)# , ha='left'
# plt.show()




