from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Download the MNIST digits dataset and take some digit
digits = fetch_mldata('MNIST original')
original_digit = digits.data[22222, :]

# Principal components to examine
n_pcs = [784, 394, 200, 100, 50, 5]
n = len(n_pcs)

for i, n_pc in enumerate(n_pcs):
	# Create a PCA model and fit it with the whole digits dataset
	pca = PCA(n_components=n_pc)
	pca.fit(digits.data)

	# Transform original digit into space of principal components
	# note: original_digit is first reshaped from 1D numpy array to 1D numpy matrix
	compressed_digit = pca.transform(original_digit.reshape(1, -1))

	# Transform back (reconstruct) image in the original space 
	reconstructed_digit = pca.inverse_transform(compressed_digit)

	# Plot the reconstructed image, first reshape to image dimensions
	plt.subplot(1, n, i + 1)
	plt.imshow(reconstructed_digit.reshape(28, 28), cmap=plt.cm.gray_r)
	plt.tick_params(which='both',  bottom='off', top='off', left='off', right='off', 
		labelbottom='off', labeltop='off', labelleft='off', labelright='off')
	plt.xlabel("PCs = " + str(n_pc))

plt.show()