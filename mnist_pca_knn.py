from __future__ import print_function

from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

digits = fetch_mldata('MNIST original')
X, y = digits.data, digits.target
N, d = X.shape

train_percentage = 0.2
test_percentage = 0.1
N_train = int(train_percentage * N)
N_test = int(test_percentage * N)
X_train, y_train = digits.data[:N_train, :], digits.target[:N_train]
X_test, y_test = digits.data[N_train:N_train+N_test, :], digits.target[N_train:N_train+N_test]

classifier = KNeighborsClassifier(n_neighbors=1)
n_pcs = np.linspace(1, d, 10).astype(int)
accuracies = []

for n_pc in n_pcs:
	pca = PCA(n_components=n_pc)
	pca.fit(X_train)
	X_train_reduced = pca.transform(X_train)
	X_test_reduced = pca.transform(X_test)

	classifier.fit(X_train_reduced, y_train)
	y_pred = classifier.predict(X_test_reduced)
	accuracy = accuracy_score(y_test, y_pred)
	accuracies.append(accuracy)
	print("For n_pc = {0:3d}, accuracy = {1}").format(n_pc, accuracy)

plt.plot(n_pcs, accuracies)
plt.xlabel("Number or PCs taken")
plt.ylabel("Accuracy with 1NN")
plt.show()