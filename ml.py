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


# kMeans clustering for unsupervized data
def kMeans(data,K,niter):
    #Put your code here
    # V represented predicted labels
    V = np.zeros(data.shape[0])
    
    # randomly select K centers
    centers = random.sample(data, K)
    
    # maintain a dictionary for centers
    clusters = {}
    
    # N x K array, column indicating data point, K indicating distance to that cluster point
    for i in range(niter):
        # matrix of distance against K-centers
        distances = np.array(dist.cdist(data,centers,'euclidean'))
        V = np.argmin(distances, axis=1)
        V = np.array(V)
        
        # store cluster in dictionary
        for i,val in enumerate(data):
            key = V[i]
            try:
                clusters[key].append(val)
            except KeyError:
                clusters[key] = [val]

        # repick centers
        new_centers = []
        keys = sorted(clusters.keys())
        for k in keys:
            new_centers.append(np.mean(clusters[k], axis=0))
        
        centers = np.array(new_centers)
        # end of an iteration
    
    return (V, centers)

# Load up some data, which we will store in a variable called problem1
print 'loading dataset...'
personalities = glob.glob('./data/*/*-pers*')
fnames = np.asarray(personalities)
N = fnames.size     # number if personalities
dim  = 22           # number of dimensions
print '... data loaded'
print 'number of personalities files = ', N

# N x 22 data array
data = np.zeros((N, dim) )
#print 'data size = ', data.shape
for idx, f in enumerate(fnames):
    with open(f) as f:
        entry = np.zeros(dim)
        item = json.load(f)

        # process JSON to np array
        category = ['personality','needs','values']
        #category = ['personality']
        i = 0

        for cat in category:
            for j, val in enumerate(item[cat]):
                entry[i] = item[cat][j]['percentile']
                i += 1

        data[idx] = entry.T
        #pprint(item['personality'])


print "data shape = ", data.shape

3
# run k-means clustering algorithm
(predicted_labels, centers) = kMeans(data, K=5, niter=100)

print predicted_labels
