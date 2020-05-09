import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# PLA
"""
Each line of the data set contains one (xn,yn) with xn ∈ R . 
The first 4 numbers of the line contains the components of xn orderly, 
the last number is yn. Please initialize your algorithm with w = 0 and take sign(0) as −1. 
As a friendly reminder, remember to add x0 = 1 as always!

6. 
(*, 20 points) Implement a version of PLA by visiting examples in fixed, 
pre-determined random cycles throughout the algorithm. 
Run the algorithm on the data set. 
Please repeat your experiment for 1126 times, 
each with a different random seed. 
What is the average number of updates before the algorithm halts? 
Plot a histogram ( https://en.wikipedia.org/wiki/Histogram ) to show the number of updates versus the frequency of the number.
"""


# An array recording the number of updates before the PLA halts
updateArr = []

# Run PLA for 1126 times
for i in range(1126):
	# Read data
	data = np.loadtxt('hw1_6_train.dat')

	# Add column x0
	x0 = np.ones(len(data))
	x0.shape = (len(data),1)
	data = np.hstack((x0, data))

	# Randomly sort the data
	np.random.shuffle(data)

	# Initialize weight vector
	w = np.zeros(5)
	numUpdate = 0

	# Run PLA
	error = 1
	while(error == 1):
		error = 0
		for j in range(len(data)):		
			sign = np.sign(np.dot(data[j][:5],w))
			if (sign == 0 or sign == -1) and (data[j][5] == 1):
				w += data[j][:5]
				error = 1
				numUpdate += 1			
			elif (sign == 1) and (data[j][5] == 0 or data[j][5] == -1):
				w -= data[j][:5]
				error = 1
				numUpdate += 1			
	updateArr.append(numUpdate)
	print('run pla: {} , number of updates: {}'.format(i, numUpdate))


# Print the average number of updates before the algorithm halts
print('Average number of updates:{:.2f}'.format(np.mean(updateArr)))
# Plot histogram to show the number of updates versus the frequency of the number.
plt.hist(updateArr, bins=30)
plt.title("Number of Updates vs Frequency")
plt.xlabel("Number of Updates(times)")
plt.ylabel("Frequency(times)")
plt.show()




















