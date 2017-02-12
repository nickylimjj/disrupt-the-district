from __future__ import division
import numpy as np
import random
from sklearn import neighbors
import scipy.spatial.distance as dist

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