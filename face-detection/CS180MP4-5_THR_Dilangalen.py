import cv2
import numpy as np
from time import time
from random import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import svm
# preprocessing

s1 = 'orl_faces/s1/'
s2 = 'orl_faces/s2/'
s3 = 'orl_faces/s3/'
s4 = 'orl_faces/s4/'
s5 = 'orl_faces/s5/'
s6 = 'orl_faces/s6/'
s7 = 'orl_faces/s7/'
s8 = 'orl_faces/s8/'
s9 = 'orl_faces/s9/'
s10 = 'orl_faces/s10/'
s11 = 'orl_faces/s11/'
s12 = 'orl_faces/s12/'
s13 = 'orl_faces/s13/'
s14 = 'orl_faces/s14/'
s15 = 'orl_faces/s15/'
s16 = 'orl_faces/s16/'
s17 = 'orl_faces/s17/'
s18 = 'orl_faces/s18/'
s19 = 'orl_faces/s19/'
s20 = 'orl_faces/s20/'
s21 = 'orl_faces/s21/'
s22 = 'orl_faces/s22/'
s23 = 'orl_faces/s23/'
s24 = 'orl_faces/s24/'
s25 = 'orl_faces/s25/'
s26 = 'orl_faces/s26/'
s27 = 'orl_faces/s27/'
s28 = 'orl_faces/s28/'
s29 = 'orl_faces/s29/'
s30 = 'orl_faces/s30/'
s31 = 'orl_faces/s31/'
s32 = 'orl_faces/s32/'
s33 = 'orl_faces/s33/'
s34 = 'orl_faces/s34/'
s35 = 'orl_faces/s35/'
s36 = 'orl_faces/s36/'
s37 = 'orl_faces/s37/'
s38 = 'orl_faces/s38/'
s39 = 'orl_faces/s39/'
s40 = 'orl_faces/s40/'

s = [	s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
		s11, s12, s13, s14, s15, s16, s17, s18, s19, s20,
		s21, s22, s23, s24, s25, s26, s27, s28, s29, s30,
		s31, s32, s33, s34, s35, s36, s37, s38, s39, s40,  ]


# initialize sample (60 % of dataset)
sample = np.array([])
labels = []
for x in range(40):
	for y in range(6):
		filename = s[x] + str(y+1) + '.pgm'
		temp = cv2.imread(filename, 0)
		temp = temp.flatten()
		temp = temp/255.0
		sample = np.concatenate((sample, temp), axis = 0)
		labels.append(s[x])

sample = sample.reshape((240, 10304))
	
# initializing test set (40% of dataset)
test = np.array([])
test_lbls = []
for x in range(40):
	for y in range(7, 11):
		filename = s[x] + str(y) + '.pgm'
		temp = cv2.imread(filename, 0)
		temp = temp/255.
		temp = temp.flatten()
		test = np.concatenate((test, temp), axis = 0)
		test_lbls.append(s[x])

test = test.reshape((160, 10304))



# Standard Scaling the training and test set
t0 = time()
scaler = StandardScaler()
scaler.fit(sample)
sample = scaler.transform(sample)
test = scaler.transform(test)
print "Done Standardization in %0.3f" %(time() - t0)


### ANN Model ###
# Computing PCA
# n_components = 10
# t0 = time()
# pca = PCA(n_components = n_components, svd_solver = 'randomized', whiten = True).fit(sample)
# eigenfaces = pca.components_.reshape((n_components, 112, 92))
# sample = pca.transform(sample)
# test = pca.transform(test)
# print "Done Computing PCA in %0.3f" %(time() - t0)

nodes = 5152			# default
training_set_scores = []
test_set_scores = []
# max_nodes = 10304		# used for case 2
max_nodes = 5153		# used for case 1
run = 1
nodes_per_run = []
while nodes < max_nodes:
	t0 = time()
	print "run: %d, nodes: %d" % (run, nodes)
	run += 1

	nodes_per_run.append(nodes)

	# set number of hidden nodes to 'nodes' or 'n_components' when applying PCA
	clf = MLPClassifier(	activation = 'logistic',
							hidden_layer_sizes = (nodes,), 
							max_iter = 200,
							alpha = 1e-5,
	             			solver = 'adam',
	             			random_state = 1,
	                    	learning_rate_init = 1e-3,
	                    	verbose = False	                    )


	clf.fit(sample, labels)
	training_set_score = clf.score(sample, labels) 
	test_set_score = clf.score(test, test_lbls)

	training_set_scores.append(training_set_score*100)
	test_set_scores.append(test_set_score*100)

	print "Training set score: %f" % training_set_score
	print "Test set score: %f" % test_set_score
	print "Done in:	%0.3f s" % (time() - t0)

	nodes = nodes + 100
	if(nodes > max_nodes):
		nodes = max_nodes

	

## plot nodes vs accuracy
# plt.plot(nodes_per_run, training_set_scores, 'b', label = "training set" )
# plt.plot(nodes_per_run, test_set_scores, 'r', label = "test set")
# plt.legend(loc = 3)
# plt.title('Hidden Layer Nodes vs Accuracy')
# plt.xlabel('nodes')
# plt.ylabel('accuracy (%)')
# plt.axis([5151.9, 10304.1 , 0, 110])
# plt.show()


### SVM Model ###
training_set_scores = []
test_set_scores = []
gamma_values = []
deg_values = []

for i in range(1):		# default
# for i in range(5):		# poly,degree
# for i in range(10):		# rbf,gamma
	t0 = time()
	print "run # %d" %(i+1)	

	clf = svm.SVC()		# default

	# deg = i+1
	# deg_values.append(deg)
	# clf = svm.SVC(kernel = 'poly', degree = i+1)

	# g = 0.1 * (i+1)
	# gamma_values.append(g)
	# clf = svm.SVC(kernel = 'rbf', gamma = g )

	t0 = time()
	clf.fit(sample, labels)
	t0 = time()
	training_set_score = clf.score(sample, labels)
	test_set_score = clf.score(test, test_lbls)

	print "Training set score: %f" % training_set_score
	print "Test set score: %f" % test_set_score
	print "Done in %0.3f" %(time() - t0)

	training_set_scores.append(training_set_score*100)
	test_set_scores.append(test_set_score*100)

### plot degree vs accuracy ###
# plt.plot(deg_values, training_set_scores, 'b', label = "training set" )
# plt.plot( deg_values, test_set_scores, 'r', label = "test set")
# plt.legend(loc = 3)
# plt.title('Polynomial degree vs Accuracy')
# plt.xlabel('degree')
# plt.ylabel('accuracy (%)')
# plt.axis([0.9, 5.1 , 0, 110])
# plt.show()

### plot gamma vs accuracy ###
# plt.plot(gamma_values, training_set_scores, 'b', label = "training set" )
# plt.plot( gamma_values, test_set_scores, 'r', label = "test set")
# plt.legend(loc = 3)
# plt.title('Gamma vs Accuracy')
# plt.xlabel('gamma')
# plt.ylabel('accuracy (%)')
# plt.axis([0.0, 1.1 , 0, 110])
# plt.show()
