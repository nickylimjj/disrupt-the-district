#!/usr/local/bin/python

from __future__ import division
import sys, getopt
import string
from twython import Twython
import json
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
from sklearn.lda import LDA
from mlcluster import kMeans
import mlclassify as ml
import time
import gendata as gd

def main(argv):
    # seed the RNG for repeatability
    np.random.seed(seed=61801) 

    # Load up some data, which we will store in a variable called problem1
    print 'loading dataset...'
    labels = glob.glob('./data/*')

    # remove label
    for index, item in enumerate(labels):
        if labels[index] == './data/Unknown':
            bad_idx = index
    labels.pop(bad_idx)

    personalities = glob.glob('./data/[!Unknown]*/*-pers*')

    fnames = np.asarray(personalities)
    labels = np.asarray(labels)
    N = fnames.size     # number if personalities
    M = labels.size    # number of labels
    dim  = 22           # number of dimensions
    print '... data loaded'
    print 'number of personalities files = ', N

    # N x (22) data array
    feat_data = np.zeros((N, dim))
    feat_names = ['0']*dim
    label_names = ['0']*M
    label_data = np.zeros((N,1))

    file_idx = 0    # 0 - (N-1)

    for i, dir in enumerate(labels):
        label_names[i] = dir.split("/")[2]
        labelled_personalities =  glob.glob(dir + '/*-pers*')
        labelled_personalities = np.asarray(labelled_personalities)

        for idx, f in enumerate (labelled_personalities):
            with open(f) as f:
                entry = np.zeros(dim)
                item = json.load(f)

                # process JSON to np array
                category = ['personality','needs','values']
                #category = ['personality']
                dim_idx = 0     # 0 - (dim-1)

                for cat in category:
                    for j, val in enumerate(item[cat]):
                        entry[dim_idx] = item[cat][j]['percentile']
                        if (file_idx == 0):
                            feat_names[dim_idx] = item[cat][j]['name']

                        dim_idx += 1


                feat_data[file_idx] = entry.T

                label_data[file_idx] = i
                file_idx += 1;
                #pprint(item['personality'])


    # Randomly permute the files we have
    idx=np.random.permutation(N)
    feat_data = feat_data[idx]
    label_data = label_data[idx]

    print "feature data shape = ", feat_data.shape
    print "label data shape = ", label_data.shape
    print "label_names\n\t", label_names
    print "feature names\n\t", feat_names

    print "feature data\n\t", feat_data[0]

    val_index = int(feat_data.shape[0] / 10 * 5)
    test_index = int(feat_data.shape[0] / 10 * 10)
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
        print algo, ': runtime ', round(end-start, 3), 's\t error = ', err*100, '%'

    start = time.time()
    predict = LDA().fit(traindata, np.ravel(trainlabels)).predict(valdata)
    err = ml.classifierError(vallabels, predict)
    end = time.time()
    print 'LDA \t: runtime', round(end-start, 3), 's error = ', err*100, '%'
    return (LDA().fit(traindata, np.ravel(trainlabels)), label_names)

# pick LDA (change to lowest err)
def ML_model(username):
    (model, labels) = main(sys.argv[1:])
    err = gd.gendata('Unknown', username)

    print err
    if (err):
        print '[*] invalid username'
        sys.exit(2)


    # GENERATE FEATURE
    filename = './data/Unknown/'+username+'-pers.json'
    dim = 22
    feature = np.zeros((dim,1))
    print filename
    with open(filename) as f:
        entry = np.zeros(dim)
        item = json.load(f)

        # process JSON to np array
        category = ['personality','needs','values']
        #category = ['personality']
        dim_idx = 0     # 0 - (dim-1)

        for cat in category:
            for j, val in enumerate(item[cat]):
                feature[dim_idx] = item[cat][j]['percentile']

    return labels[int(model.predict(feature.T)[0])]

if __name__ == "__main__":
    main(sys.argv[1:])