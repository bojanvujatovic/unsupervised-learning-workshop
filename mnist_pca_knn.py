from __future__ import print_function

from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Download the MNIST digits dataset and store in the variables
digits = fetch_mldata('MNIST original')
X, y = digits.data, digits.target
N, d = X.shape

# Take only part of dataset for training and testing for faster execution
train_percentage = 0.2
test_percentage = 0.1
N_train = int(train_percentage * N)
N_test = int(test_percentage * N)
X_train, y_train = digits.data[:N_train, :], digits.target[:N_train]
X_test, y_test = digits.data[N_train:N_train+N_test, :], digits.target[N_train:N_train+N_test]

# Create a classifier
classifier = KNeighborsClassifier(n_neighbors=1)

# Create a list of number of principal components to examine
# note: they have to be be integers so cast floats to ints
n_pcs = np.linspace(1, d, 10).astype(int)

accuracies = []
for n_pc in n_pcs:
	# Create a PCA model and fit it with training data
	pca = PCA(n_components=n_pc)
	pca.fit(X_train)

	# Reduce the training and the test data according to fitted model
	X_train_reduced = pca.transform(X_train)
	X_test_reduced = pca.transform(X_test)

	# Fit the classifier with reduced training data
	classifier.fit(X_train_reduced, y_train)

	# Make predictions
	y_pred = classifier.predict(X_test_reduced)
	
	# Calculate the accuracy on the test set
	accuracy = accuracy_score(y_test, y_pred)
	accuracies.append(accuracy)
	print("For n_pc = {0:3d}, accuracy = {1}").format(n_pc, accuracy)

# Plot the accuracy of the classifier on test set versus number of principal components
plt.plot(n_pcs, accuracies)
plt.xlabel("Number or PCs taken")
plt.ylabel("Accuracy with 1NN")
plt.show()