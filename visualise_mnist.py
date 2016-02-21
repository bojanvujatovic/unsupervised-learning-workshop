from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

digits = fetch_mldata('MNIST original')
original_digit = digits.data[22222, :]

n_pcs = [784, 394, 200, 100, 50, 5]
n = len(n_pcs)

for i, n_pc in enumerate(n_pcs):
	pca = PCA(n_components=n_pc)
	pca.fit(digits.data)
	compressed_digit = pca.transform(original_digit.reshape(1, -1))
	reconstructed_digit = pca.inverse_transform(compressed_digit.reshape(1, -1))
	reconstructed_digit.shape = (28, 28)

	plt.subplot(1, n, i + 1)
	plt.imshow(reconstructed_digit, cmap=plt.cm.gray_r)
	plt.tick_params(which='both',  bottom='off', top='off', left='off', right='off', 
		labelbottom='off', labeltop='off', labelleft='off', labelright='off')
	plt.xlabel("PCs = " + str(n_pc))

plt.show()