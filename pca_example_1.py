from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np

digits = fetch_mldata('MNIST original')
X, y = digits.data, digits.target
N, d = X.shape

train_percentage = 0.8
N_train = int(train_percentage * N)
N_test = N - N_train
X_train, y_train = digits.data[:N_train, :], digits.target[:N_train]
X_test, y_test = digits.data[N_train:, :], digits.target[N_train:]

knn = KNeighborsClassifier(n_neighbors=5)
n_pcs = np.linspace(1, d, 50).astype(int)

for n_pc in n_pcs:
	pca = PCA(n_components=n_pc)
	pca.fit(X_train)
	X_train_reduced = pca.transform(X_train)
	X_test_reduced = pca.transform(X_test)

	knn.fit(X_train_reduced, y_train)
	y_pred = knn.predict(X_test_reduced)
	accuracy = accuracy_score(y_test, y_pred)

	print(n_pc, accuracy)