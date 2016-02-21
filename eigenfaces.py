from __future__ import print_function

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


lfw_people = fetch_lfw_people(min_faces_per_person=100, resize=0.4)
X = lfw_people.data

n, height, width = lfw_people.images.shape
d = height * width

pca = PCA().fit(X)
eigenfaces = pca.components_.reshape((n, height, width))

plt.imshow(eigenfaces[3].reshape((height, width)), cmap=plt.cm.gray)
plt.show()

