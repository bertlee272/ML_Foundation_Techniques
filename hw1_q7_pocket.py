import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Pocket

"""
Next, we play with the pocket algorithm. 
Modify your PLA in the previous problem by adding the ‘pocket’ steps to the algorithm. 
We will use http://www.csie.ntu.edu.tw/~htlin/course/mlfound19fall/hw1/hw1_7_train.dat
as the training data set D, and http://www.csie.ntu.edu.tw/~htlin/course/mlfound19fall/hw1/hw1_7_test.dat
as the test set for “verifying” the g returned by your algorithm (see lecture 4 about verifying). 
The sets are of the same format as the previous one.

7. (*, 20 points) Run the pocket algorithm with a total of 100 updates on D, 
and verify the performance of wpocket using the test set. 
Please repeat your experiment for 1126 times, each with a different random seed. 
What is the average error rate on the test set? Plot a histogram to show the error rate versus frequency.
"""

# Count the number of error for a specific weight vector in all the data
def countWrong(w, data):
	wrongTime = 0
	for i in range(len(data)):
		sign = np.sign(np.dot(data[i][:5],w))
		if ((sign == 0 or sign == -1) and (data[i][5] == 1)) or ((sign == 1) and (data[i][5] == 0 or data[i][5] == -1)):
			wrongTime += 1
	return wrongTime

# Find an error and adjust the weight vector 
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

	# An array recording the error rate 
	errorRateArr = []

	# Read data
	dataTrain = np.loadtxt('hw1_7_train.dat')
	dataTest = np.loadtxt('hw1_7_test.dat')

	# Add column x0 for training data
	x0 = np.ones(len(dataTrain))
	x0.shape = (len(dataTrain),1)
	dataTrain = np.hstack((x0, dataTrain))

	# Add column x0 for testing data
	x0_test = np.ones(len(dataTest))
	x0_test.shape = (len(dataTest),1)
	dataTest = np.hstack((x0_test, dataTest))

	# Run pocket algorithm for 1126 times
	for i in range(1126):
		# randomly sort the data
		np.random.shuffle(dataTrain)
		# initialize the weight vector
		w = np.zeros(5)
		# initialize the pocket weight vector and the least number of error
		w_pocket = np.zeros(5)
		leastWrong = countWrong(w_pocket, dataTrain)

		numUpdate = 0
		# Run the pocket algorithm with a total of 100 updates
		while (numUpdate < 100):
			w_new, altered = findAndCorrect(np.copy(w), dataTrain)
			if altered == 0:
				print('perfect W')
				break
			else:
				w = np.copy(w_new)
				numUpdate += 1

			wrong = countWrong(w, dataTrain)
			if wrong < leastWrong:
				leastWrong = wrong
				w_pocket = np.copy(w)		

		# count the error using the pocket weight vector(least error) and record it
		wrongTest = countWrong(w_pocket, dataTest)
		errorRateArr.append(wrongTest/len(dataTest)*100)
		print('run pocket: {} , error rate: {:.1f}%'.format(i, wrongTest/len(dataTest)*100))

	print('Average error rate: {:.1f}%'.format(np.mean(errorRateArr)))
	# Plot histogram to show the error rate versus the frequency of the number.
	plt.hist(errorRateArr, bins=30)
	plt.title("Error Rate vs Frequency")
	plt.xlabel("Error Rate (%)")
	plt.ylabel("Frequency (times)")	
	plt.show()





		














