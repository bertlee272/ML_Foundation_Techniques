import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# generate an (x, y) set of size N
def genData(num,noise):
	x = np.sort(np.random.uniform(-1, 1, size=num))
	y = np.sign(x)
	# generate noise for y
	noiseIdx = np.random.uniform(0, 1, size=num)
	for i in range(len(noiseIdx)):
		if noiseIdx[i] < noise:
			y[i] = y[i] * -1
	return x, y

# generate an theta array of size N+1
def genThetaArr(dataSize, x):
	thetaArr = np.zeros(dataSize+1)
	# set the first theta as a random number less than any x
	thetaArr[0] = x[0] - 1
	# set the last theta as a random number bigger than any x
	thetaArr[-1] = x[-1] + 1
	# set other thetas as the median of adjacent x
	for i in range(1, dataSize):
		thetaArr[i] = (x[i-1]+x[i])/2
	return thetaArr

# calculate Ein , given x, y, s
def calculateEin(x, y, thetaArr, s):
	sortIdx = x.argsort()
	x = np.sort(x)
	y = y[sortIdx]	
	errorArr = []
	for i in range(len(thetaArr)):
		error = 0
		for j in range(len(x)):
			hx = s * np.sign(x[j] - thetaArr[i])
			if hx != y[j]:
				error += 1
		errorArr.append(error)
	best_theta = thetaArr[np.argmin(errorArr)]
	min_Ein = np.min(errorArr)
	return min_Ein/len(x), best_theta

def decisionStump(x, y, thetaArr):
	# calculate Ein when s = 1
	Ein1, theta1 = calculateEin(x, y, thetaArr, 1)
	# calculate Ein when s = -1
	Ein_1, theta_1 =calculateEin(x, y, thetaArr, -1)
	if Ein1 < Ein_1:
		return Ein1, 1, theta1
	else:
		return Ein_1, -1, theta_1

# -------------------------
if __name__ == '__main__':
	dataSize = 20
	noiseRate = 0.2
	iter = 1000
	EinArr = []
	EoutArr = []
	EinOutArr = []
	for i in range(iter):
		print('-------------------------iter:', i)
		x, y = genData(dataSize,noiseRate)
		thetaArr = genThetaArr(dataSize, x)
		Ein, s, theta = decisionStump(x, y, thetaArr)
		Eout = 0.5 + 0.3 * s * (np.abs(theta) - 1)
		EinArr.append(Ein)
		EoutArr.append(Eout)
		EinOutArr.append(Ein-Eout)
	print('Average Ein rate: {:.3f}%'.format(np.mean(EinArr)))
	print('Average Eout rate: {:.3f}%'.format(np.mean(EoutArr)))
	print('Average Ein - Eout: {:.3f}%'.format(np.mean(EinOutArr)))

	plt.hist(EinOutArr, bins=30)
	plt.title("Ein-Eout vs Frequency")
	plt.xlabel("Ein-Eout(%)")
	plt.ylabel("Frequency(times)")
	plt.show()











