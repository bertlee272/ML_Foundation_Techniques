import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Pocket

"""
8. (*, 20 points) Modify your algorithm in the previous problem to return w100 
(the PLA vector after 100 updates) instead of w藛 (the pocket vector) after 100 updates. 
Run the modified algorithm on D, and verify the performance using the test set. 
Please repeat your experiment for 1126 times, each with a different random seed. 
What is the average error rate on the test set? 
Plot a histogram to show the error rate versus frequency. 
Compare your result to the previous problem and briefly discuss your findings.
"""

# count the number of error in all the data
def countWrong(w, data):
	wrongTime = 0
	for i in range(len(data)):
		sign = np.sign(np.dot(data[i][:5],w))
		if ((sign == 0 or sign == -1) and (data[i][5] == 1)) or ((sign == 1) and (data[i][5] == 0 or data[i][5] == -1)):
			wrongTime += 1
	return wrongTime

# find an error and adjust the weight vector 
def findAndCorrect(w, data):
	altered = 0
	# Be aware that we are looking for a "random" mistake
	np.random.shuffle(data)	
	for i in range(len(data)):
		sign = np.sign(np.dot(data[i][:5],w))
		if (sign == 0 or sign == -1) and (data[i][5] == 1):
			w += data[i][:5]
			altered = 1
			break
		elif (sign == 1) and (data[i][5] == 0 or data[i][5] == -1):
			w -= data[i][:5]
			altered = 1
			break
	return w, altered


if __name__ == '__main__':

	# an array recording the error rate 
	errorRateArr = []

	# read data
	dataTrain = np.loadtxt('hw1_7_train.dat')
	dataTest = np.loadtxt('hw1_7_test.dat')

	# add column x0 for training data
	x0 = np.ones(len(dataTrain))
	x0.shape = (len(dataTrain),1)
	dataTrain = np.hstack((x0, dataTrain))

	# add column x0 for testing data
	x0_test = np.ones(len(dataTest))
	x0_test.shape = (len(dataTest),1)
	dataTest = np.hstack((x0_test, dataTest))

	# run pocket algorithm for 1126 times
	for i in range(1126):
		# randomly sort the data
		np.random.shuffle(dataTrain)
		# initialize the weight vector
		w = np.zeros(5)

		numUpdate = 0
		# Run the pocket algorithm with a total of 100 updates without updating the best weight vector
		while (numUpdate < 100):
			w_new, altered = findAndCorrect(np.copy(w), dataTrain)
			if altered == 0:
				print('perfect W')
				break
			else:
				w = np.copy(w_new)
				numUpdate += 1

		# count the error using the weight vector after 100 updates and record it
		wrongTest = countWrong(w, dataTest)
		errorRateArr.append(wrongTest/len(dataTest)*100)
		print('run pocket: {} , error rate: {:.1f}%'.format(i, wrongTest/len(dataTest)*100))


	print('Average error rate: {:.1f}%'.format(np.mean(errorRateArr)))
	plt.hist(errorRateArr, bins=30)
	plt.title("Error Rate vs Frequency")
	plt.xlabel("Error Rate (%)")
	plt.ylabel("Frequency (times)")	
	plt.show()




		














