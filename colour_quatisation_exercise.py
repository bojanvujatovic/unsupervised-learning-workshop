# Authors: Robert Layton <robertlayton@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD 3 clause

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

# Number of clusters - reduced colours
k = 64

# Load the Summer Palace photo
china = load_sample_image("china.jpg")

# Dividing by 255 is important so that plt.imshow behaves works well on float data 
# (need to be in the range [0-1])
china = np.array(china) / 255.0

# Load Image and transform to a 2D numpy array.
width, height, channels = tuple(china.shape)
image_array = np.reshape(china, (width * height, channels))

# TODO: initialise KMeans with k clusters

# TODO: fit KMeans with image colours OR sample from image colours (faster)

# TODO: get the cluster labels from KMeans

def recreate_image(image, codebook, labels, width, height):
    """Recreate the (compressed) image from the code book & labels"""
    channels = codebook.shape[1]
    image = np.zeros((width, height, channels))
    label_idx = 0
    for i in range(width):
        for j in range(height):
            # TODO: Assign the right colour
            image[i][j] = None
            label_idx += 1
    return image

# Display all results, alongside original image
plt.figure(figsize=(16,6))

plt.subplot(1, 2, 1)
plt.title('Original image (96,615 colours)')
plt.imshow(china)
plt.tick_params(which='both',  bottom='off', top='off', left='off', right='off', 
    labelbottom='off', labeltop='off', labelleft='off', labelright='off')

plt.subplot(1, 2, 2)
plt.title('Quantised image (' + str(k) + ' colours, K-Means)')
plt.imshow(recreate_image(china, kmeans.cluster_centers_, labels, width, height))
plt.tick_params(which='both',  bottom='off', top='off', left='off', right='off', 
    labelbottom='off', labeltop='off', labelleft='off', labelright='off')

plt.show()
