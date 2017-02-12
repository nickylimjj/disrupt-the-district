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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mlcluster import kMeans
import mlclassify as ml
import time
import gendata as gd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import BernoulliNB 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.svm import LinearSVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def main(argv):
    # seed the RNG for repeatability
    np.random.seed(seed=14282357) 

    # Load up some data, which we will store in a variable called problem1
    print 'loading dataset...'
    labels = glob.glob('./data/[!Unknown]*')
    personalities = glob.glob('./data/[!Unknown]*/*-pers*')

    fnames = np.asarray(personalities)
    labels = np.asarray(labels)
    N = fnames.size     # number if personalities
    M = labels.size    # number of labels
    dim  = 22           # number of dimensions
    print '... data loaded'
    print 'number of personalities files = ', N
    print 'number of cities = ', M

    # N x (22) data array
    feat_data = np.zeros((N, dim))
    label_data = np.zeros((N,1))

    data_person_names = ['0']*N
    feat_names = ['0']*dim
    label_names = ['0']*M
    label_avg = np.zeros((M,dim))
    label_std = np.zeros((M,dim))
    
    file_idx = 0    # 0 - (N-1)

    for i, dir in enumerate(labels):
        label_names[i] = dir.split("/")[2]
        labelled_personalities =  glob.glob(dir + '/*-pers*')
        labelled_personalities = np.asarray(labelled_personalities)

        start_file_idx = file_idx
        for idx, f in enumerate (labelled_personalities):
            # get feature name of each feature
            name = f.split("/")[3].split("-")[0][1:]
            data_person_names[file_idx] = name

            # each personalities file
            with open(f) as f:
                entry = np.zeros(dim)
                item = json.load(f)

                # process JSON to np array
                category = ['personality','needs','values']
                dim_idx = 0     # 0 - (dim-1)

                # over dimension
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
        label_avg[i] = feat_data[start_file_idx:file_idx,:].mean(axis=0).T
        label_std[i] = feat_data[start_file_idx:file_idx,:].std(axis=0).T

    # get average of each city
    # Randomly permute the files we have
    idx=np.random.permutation(N)
    feat_data = feat_data[idx]
    label_data = label_data[idx]

    # calculate frequency of each label
    for i in range(len(label_names)):
        print label_names[i],"=",(label_data == i).sum()

    print "feature data shape = ", feat_data.shape
    print "label data shape = ", label_data.shape
    print "label_names\n\t", label_names
    print "feature names\n\t", feat_names
    print "label averages\n\t", label_avg
    print "label std\n\t", label_std

    # break data into sets ()
    val_index = int(feat_data.shape[0] / 10 * 7.5)
    test_index = int(feat_data.shape[0] / 10 * 10)
    
    traindata = feat_data[:val_index]
    trainlabels = label_data[:val_index]
    print 'trainlabels', trainlabels.T

    valdata = feat_data[val_index:test_index]
    vallabels = label_data[val_index:test_index]
    print 'vallabels', vallabels.T

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

    # run classifiers
    errors = []


    funcdict = {
    'LDA': LDA,
    'BernoulliNB': BernoulliNB,
    'MultinomialNB': MultinomialNB,
    'LinearSVC': LinearSVC,
    'LogisticRegression': LogisticRegression,
    'KNeighborsClassifier': KNeighborsClassifier,
    'DecisionTreeClassifier': DecisionTreeClassifier
    }

    for key in funcdict:
        start = time.time()
        predicted_labels = funcdict[key]().fit(traindata, np.ravel(trainlabels)).predict(valdata)
        end = time.time()
        err = ml.classifierError(vallabels, predicted_labels)
        errors.append(err)

        print key, ': runtime', round(end-start, 3), 's error = ', err*100, '%'
        print '----'
   
    # pick the one with lowest error
    chosen = np.argmin(errors)
    func = [
    LDA,
    BernoulliNB,
    MultinomialNB,
    LinearSVC,
    LogisticRegression,
    KNeighborsClassifier,
    DecisionTreeClassifier
    ]

    print 'we choose',func[chosen],'with err = ', errors[chosen]*100,'%'

    return (func[chosen]().fit(traindata, np.ravel(trainlabels)), 
        label_names,
        label_avg,
        label_std)


# FUNCTION FOR mlmatch.py
def ML_model(username):
    (model_cfy, labels_city, label_avg, label_std) = main(sys.argv[1:])
    gd.gendata('Unknown', username)

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
                dim_idx += 1

    print 'feature\n\t:',feature.T
    idx = int(model_cfy.predict(feature.T)[0])
    distances = np.array(dist.cdist(feature.T,label_avg,'euclidean'))
    V = np.argsort(distances)
    print V[0]
    print distances
    rank = ['0']*len(labels_city)
    
    for i, val in enumerate(rank):
        rank[i] = labels_city[V[0,i]]
    print rank

    ret_dict = {}
    ret_dict['handle'] = username
    ret_dict['mlrecommends'] = labels_city[idx]
    ret_dict['user-signature'] = feature.T[0].tolist()
    ret_dict['similarity'] = {}
    for i, val in enumerate(rank):
        ret_dict['similarity'][i] = {
            'city': rank[i], 
            'similarity-score' : distances[0,V[0,i]],
            'city-signature' : label_avg[idx].tolist()
            }

    print '>>>>>>>>>>>>>>>>>'
    ret_json = json.dumps(ret_dict,sort_keys=True)
    pprint(ret_json)
    return (ret_json)


if __name__ == "__main__":
    main(sys.argv[1:])