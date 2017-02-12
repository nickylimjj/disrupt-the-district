from __future__ import division
import numpy as np
import random
from sklearn import neighbors
import scipy.spatial.distance as dist
from scipy import stats

def bayesClassifier(data,pi,means,cov):
    #print means.shape, np.dot(means.T, numpy.linalg.inv(cov)).T.shape
    #print np.log(pi).shape, np.dot(data, means.transpose().dot(np.linalg.inv(cov)).T).shape, \
    #                 np.diag(0.5*means.transpose().dot(np.linalg.inv(cov)).dot(means)).shape
    return np.argmax(np.log(pi) + \
                     np.dot(data, means.transpose().dot(numpy.linalg.inv(cov)).T) - \
                     np.diag(0.5*means.transpose().dot(np.linalg.inv(cov)).dot(means))
                     , axis = 1) # maximizing over y = {0,1,...,M=2}


def classifierError(truelabels,estimatedlabels):
    return (truelabels != estimatedlabels).sum() / estimatedlabels.size

def trainLDA(trainfeat,trainlabel):
    nlabels=int(trainlabel.max())+1 #Assuming all labels up to nlabels exist.
    pi=np.zeros(nlabels) # store your prior in here
    means=np.zeros((nlabels,trainfeat.shape[1])) # store the class means in here
    cov=np.zeros((trainfeat.shape[1],trainfeat.shape[1])) # store the covariance matrix in here
    N = trainfeat.shape[0]
    # Put your code here
    for i in range(nlabels):
        pi[i] = trainfeat[trainlabel==i].shape[0] / N
        means[i,:] = np.sum(trainfeat[trainlabel==i],axis=0) / trainfeat[trainlabel==i].shape[0]
        
    # calculate estimated cov
    for i in range(N):
        cov += np.outer((trainfeat[i]- means[trainlabel[i]]),(trainfeat[i]- means[trainlabel[i]]))
    
    cov /= N - nlabels
    return (pi,means,cov)

def LDAclassifier(trainingdata, traininglabel, data):
    # estimate priors, means and cov matrix based on training data
    pi,means,cov = trainLDA(trainingdata,traininglabels)
    return bayesClassifier(data, pi, means.T, cov), pi, means, cov

def kNN(trainfeat,trainlabel,testfeat, k):
    #Put your code here
    V = np.zeros(testfeat.shape[0])
    distances = np.array(dist.cdist(testfeat,trainfeat,'euclidean'))
    closest = np.argpartition(distances,k)[:,0:k]
    for i in range(testfeat.shape[0]):
        V[i] = stats.mode(trainlabel[closest[i]])[0]
    return V