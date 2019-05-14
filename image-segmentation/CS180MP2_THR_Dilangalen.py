import cv2
from sklearn.cluster import KMeans
import numpy as np
import os
from time import time

curr_dir = os.path.abspath(os.path.dirname(__file__))
filaria_dir = os.path.join(curr_dir, 'filarioidea')			#filariodea directory
plasmodia_dir = os.path.join(curr_dir, 'plasmodium')		#plasmodium directory
schistosoma_dir = os.path.join(curr_dir, 'schistosoma')		#schistosoma directory

specimen = [filaria_dir, plasmodia_dir, schistosoma_dir]

# load helper code
def load_image(filename, i):
	print "Loading image: '%s' " % filename
	img_path = os.path.join(specimen[i], filename)
	return cv2.imread(img_path)

def recreate_image(original_image, center, labels, w, h):
	res = center[labels.flatten()]
	output = res.reshape((original_image.shape))
	return output

# specimen filenames
filaria = ['filaria.jpg', 'filaria2.jpg',  'filaria3.jpg', 'filaria4.jpg', 'filaria5.jpg', 'filaria6.jpg', 'filaria7.jpg', 'filaria8.jpg', 'filaria9.jpg', 'filaria10.jpg']
plasmodia = ['1c.JPG', '3c.JPG', '6c.JPG', '7c.JPG', '11c.JPG', '19c.JPG', '55c.JPG', '79c.JPG', '94c.JPG', '105c.JPG']
schistosoma = ['schistosoma.jpg', 'schistosoma2.jpg', 'schistosoma3.jpg', 'schistosoma4.jpg', 'schistosoma5.jpg', 'schistosoma6.jpg', 'schistosoma7.jpg', 'schistosoma8.jpg', 'schistosoma9.jpg', 'schistosoma10.jpg',]

# load training images
# test1 = load_image(filaria[1], 0)
# test1 = load_image(plasmodia[1], 1)
test1 = load_image(schistosoma[4], 2)
# test2 = load_image(filaria[0], 0)
# test2 = load_image(plasmodia[6], 1)
test2 = load_image(schistosoma[5], 2)

# changing colorspaces

# test1 =cv2.cvtColor(test1, cv2.COLOR_BGR2HSV)
# test1 = cv2.cvtColor(test1, cv2.COLOR_BGR2Lab)
# test2 = cv2.cvtColor(test2, cv2.COLOR_BGR2HSV)
# test2 = cv2.cvtColor(test2, cv2.COLOR_BGR2Lab)

# manually select centroids
# # selected filaria centers (2+1)
# black = test2[1,1]
# parasite = test2[1653,1067]
# others = test2[935,815]
# centroids = np.array([black, parasite, others])

# # selected plasmodia centers (2)
# parasite = test1[2325, 1660]
# others = test1[1857, 2192]
# centroids = np.array([parasite, others])

# selected schistosoma centers (2+1)
black = test2[1,1]
others = test2[1851,1458]
parasite = test2[1437,1568]
centroids = np.array([black, parasite, others])

print centroids
print centroids.shape

# rescale images
test1 = cv2.resize(test1, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
test2 = cv2.resize(test2, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)

# reshape images
test1 = test1.reshape((-1,3))
test2 = test2.reshape((-1,3))

# KMeans 
t0 = time()
print "Fitting the model..."

kmeans = KMeans(n_clusters = 4, init = 'random' , random_state = 0)
# kmeans = KMeans(n_clusters = 3, init = centroids ,n_init = 1, random_state = 0)
kmeans.fit(np.concatenate((test2, test1)))
centers = kmeans.cluster_centers_
print "Done in %0.3f s." % (time()-t0)

# load image to test, resize, reshape
print "Predicting color indices on the full image"
t0 = time()
for i in range(len(filaria)):
	# test3 = load_image(filaria[i], 0)
	# test3 = load_image(plasmodia[i], 1)
	test3 = load_image(schistosoma[i], 2)
	# test3 = cv2.cvtColor(test3, cv2.COLOR_BGR2HSV)
	# test3 = cv2.cvtColor(test3, cv2.COLOR_BGR2Lab)
	test3 = cv2.resize(test3, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
	w,h,d = test3.shape
	data = test3.reshape((-1, 3))
	# print test3.shape

	# Getting the labels of all pixels
	labels = kmeans.predict(data)
	segmented = recreate_image(test3, centers, labels, w, h)

	# saving image
	cv2.imwrite('%d.jpg' %i, segmented )
print "done in %0.3f s." %(time()-t0)
