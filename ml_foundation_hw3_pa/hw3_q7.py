import numpy as np
from scipy.special import softmax
from collections import Counter
import matplotlib.pyplot as plt
import math
from scipy.linalg import norm


# Logistic Regression
"""
19.
Implement the fixed learning rate gradient descent algorithm for logistic regression. 
Run the algorithm with η=0.01 and T=2000, 
what is Eout(g) from your algorithm, evaluated using the 0/1 error on the test set?

20.
Implement the fixed learning rate stochastic gradient descent algorithm for logistic regression. 
Instead of randomly choosing n in each iteration, please simply pick the example with the cyclic order n=1,2,…,N,1,2,…
Run the algorithm with η=0.001 and T=2000. 
What is Eout(g) from your algorithm, evaluated using the 0/1 error on the test set?

7. (20 points, *) For Questions 19 and 20 of Homework 3 on Coursera, plot a figure that shows Ein(wt)
as a function of t for both the gradient descent version and the stochastic gradient descent version
on the same figure. Describe your findings. Please print out the figure for grading.


"""
def sigmoid(x):
	return 1 / (1 + math.exp(-x))


def logisticRegressionGD(x, y, eta, t, w):
	# compute gradient Ein and Ein
	N = len(dataTrain)
	gradientEinSum = np.zeros(x.shape[1])
	gradientEin = np.zeros(x.shape[1])
	EinSum = 0
	Ein = 0
	for i in range(N):
		gradientEinSum += sigmoid(-y[i]*np.dot(w,x[i]))*(-y[i]*x[i])
		EinSum += np.log(1+np.exp(-y[i]*np.dot(w,x[i])))
	gradientEin = gradientEinSum/N
	Ein = EinSum/N
	print(t, Ein)
	# update weight vector
	w -= eta * gradientEin
	return w, Ein

def logisticRegrssionSGD(x, y, eta, t, w):
	gradientEin = np.zeros(x.shape[1])
	Ein = np.zeros(x.shape[1])
	# compute gradient Ein and Ein
	i = t % len(y)
	gradientEin = sigmoid(-y[i]*np.dot(w,x[i]))*(-y[i]*x[i])
	Ein = np.log(1+np.exp(-y[i]*np.dot(w,x[i])))
	print(t, Ein)
	# update weight vector
	w -= eta * gradientEin
	return w, Ein

def calculateError(x, y, w):
	# calculate prediction accuracy
	testSize = len(y)
	yPredict = np.zeros(testSize)	
	error = 0
	for i in range(testSize):
		yPredict[i] = np.dot(w, x[i])
		if yPredict[i] > 0 and y[i] == -1:
			error += 1
		elif yPredict[i] < 0 and y[i] == 1:
			error += 1
	errorRate = error/testSize
	return errorRate

# -------------------------
if __name__ == '__main__':
	# Read training and tesing data 
	dataTrain = np.loadtxt('hw3_train.dat')
	N, dim = dataTrain.shape
	xTrain = dataTrain[:,:-1]
	xTrain = np.insert(xTrain, 0, 0, axis=1)
	yTrain = dataTrain[:,-1]	

	dataTest = np.loadtxt('hw3_test.dat')
	xTest = dataTest[:,:-1]
	xTest = np.insert(dataTest[:,:-1],0,0,axis=1)
	yTest = dataTest[:,-1]	


	# training parameters
	T = 2000
	eta = 0.001

	Ein01ArrGD = []
	Ein01ArrSGD = []
	EinArrGD = []
	EinArrSGD = []	
	# Initialize weight vector
	w = np.zeros(dim)	

	for t in range(T):
		wGD, EinGD = logisticRegressionGD(xTrain, yTrain, eta, t, w)
		errorGD = calculateError(xTrain, yTrain, wGD)
		Ein01ArrGD.append(errorGD)
		EinArrGD.append(EinGD)
		print('-------------------------')
		print('update t: {}, GD error: {:.1f}%'.format(t+1, errorGD*100))

	w = np.zeros(dim)	
	for t in range(T):
		wSGD, EinSGD = logisticRegrssionSGD(xTrain, yTrain, eta, t, w)
		errorSGD = calculateError(xTrain, yTrain, wSGD)
		Ein01ArrSGD.append(errorSGD)
		EinArrSGD.append(EinSGD)
		print('-------------------------')
		print('update t: {}, SGD error: {:.1f}%'.format(t+1, errorSGD*100))

	t = list(range(0,T))
	f1 = plt.figure()
	plt.plot(t, Ein01ArrSGD, label='Stochastic Gradient Descent')
	plt.plot(t, Ein01ArrGD, label='Gradient Descent')
	plt.title("Ein(wt)(0/1 error) vs t")
	plt.xlabel("t")
	plt.ylabel("Ein(wt)(0/1 error)")
	plt.legend(loc='lower left')
	f1.show()

	f2 = plt.figure()
	plt.plot(t, EinArrSGD, label='Stochastic Gradient Descent')
	plt.plot(t, EinArrGD, label='Gradient Descent')
	plt.title("Ein(wt) vs t")
	plt.xlabel("t")
	plt.ylabel("Ein(wt)")
	plt.legend(loc='lower left')
	plt.show()















