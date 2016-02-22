from __future__ import print_function

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from math import ceil
from math import sqrt
import matplotlib.pyplot as plt


lfw_people = fetch_lfw_people(min_faces_per_person=10, resize=0.4)
X = lfw_people.data
n_pc = 16

n, height, width = lfw_people.images.shape
d = height * width

# Create PCA model with n_pc principal dimensions
pca = PCA(n_components=n_pc)

# Apply PCA to the data X
pca.fit(X)

# Access principal components
eigenfaces = pca.components_

# Make square grid
plot_width = int(ceil(sqrt(n_pc)))
for i in range(n_pc):
	plt.subplot(plot_width, plot_width, i + 1)

	# Reshape ith principal component and plot it
	plt.imshow(eigenfaces[i].reshape(height, width), cmap=plt.cm.gray)
	
	plt.tick_params(which='both',  bottom='off', top='off', left='off', right='off', 
		labelbottom='off', labeltop='off', labelleft='off', labelright='off')

plt.show()

