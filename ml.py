#!/usr/local/bin/python

from __future__ import division
import numpy as np
from sklearn import neighbors
from numpy import genfromtxt
import scipy.spatial.distance as dist
import random
from sklearn.cluster import KMeans
from PIL import Image
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split
import glob
import json
from pprint import pprint

from mlcluster import kMeans
import mlclassify as ml
import time


# kMeans clustering for unsupervized data


# Load up some data, which we will store in a variable called problem1
print 'loading dataset...'
personalities = glob.glob('./data/*/*-pers*')
labels = glob.glob('./data/*/label.txt')
fnames = np.asarray(personalities)
labels = np.asarray(labels)
N = fnames.size     # number if personalities
dim  = 22           # number of dimensions
print '... data loaded'
print 'number of personalities files = ', N

# N x 22 data array
feat_data = np.zeros((N, dim))
label_data = np.zeros((N,1))
labels_dict = {}
dict_idx = 0

for idx, f in enumerate(fnames):
    with open(f) as f:
        with open(labels[idx]) as l:
            entry = np.zeros(dim)
            item = json.load(f)
            label = l.read()

            # process JSON to np array
            category = ['personality','needs','values']
            #category = ['personality']
            i = 0

            for cat in category:
                for j, val in enumerate(item[cat]):
                    entry[i] = item[cat][j]['percentile']
                    i += 1

            feat_data[idx] = entry.T
            try:
                # label data
                label_data[idx] = labels_dict[label]
            except KeyError:
                # create entry
                labels_dict[label] = dict_idx
                # label data
                label_data[idx] = labels_dict[label]
                dict_idx += 1
            #pprint(item['personality'])


print "feature data shape = ", feat_data.shape
print "label data shape = ", label_data.shape

val_index = int(feat_data.shape[0] / 10 * 6)
test_index = int(feat_data.shape[0] / 10 * 8)
# break data into sets (60-20-20)
traindata = feat_data[:val_index]
trainlabels = label_data[:val_index]

valdata = feat_data[val_index:test_index]
vallabels = label_data[val_index:test_index]

testdata = feat_data[test_index:]
testlabels = label_data[test_index:]

# asset label and data have same sizes
assert traindata.shape[0] == trainlabels.shape[0]
assert valdata.shape[0] == vallabels.shape[0]
assert testdata.shape[0] == testlabels.shape[0]

print "\n"
print "train data size = ", traindata.shape[0], \
    '(', round(traindata.shape[0] / N * 100, 0), '% )'
print "val data size = ", valdata.shape[0], \
    '(',round(valdata.shape[0] / N * 100, 0), '% )'
print "test data size = ", testdata.shape[0], \
    '(',round(testdata.shape[0] / N * 100, 0), '% )'

# run k-means clustering algorithm
# (predicted_labels, centers) = kMeans(data, K=5, niter=100)

for algo in ['brute','ball_tree','kd_tree']:
    start = time.time()
    predict = neighbors.KNeighborsClassifier(algorithm=algo, p=2).fit(traindata, np.ravel(trainlabels)).predict(valdata)
    err = ml.classifierError(vallabels, predict)
    end = time.time()
    print algo, ': runtime ', end-start, 's\t error = ', err
