import numpy as np
from matplotlib import pyplot as plt

#Define a function to update the centroids
def update_centroids(X,k,labels):
    ncol = X.shape[1]
    centroids = np.zeros(shape = (k,ncol))
    for cluster in range(k):
        centroids[cluster,:] = np.mean(X[labels == cluster,:],axis =0)
    return(centroids)


#Compute the manhatan distence of two given datapoints

def manhatan_distance(point1,point2):
    import numpy as np
    dist = 0
    for i in range(len(point1)):
        dist = dist + abs(point1[i] - point2[i])
    return(dist)
    

def compute_distances_from_centroids_with_manhatan(X, centroids):  
    nrow = X.shape[0]
    manh_dist = np.zeros(shape=(nrow,centroids.shape[0]))
    for i, centroid in enumerate(centroids):
        #a = np.apply_along_axis(mahalanobis_distance, axis=1, arr = X, point2 = centroid)
        manh_dist[:,i] = np.apply_along_axis(manhatan_distance, axis=1, arr = X, point2 = centroid)
    labels = np.argmin(manh_dist, axis = 1)
    #return(eudist)
    return(labels)

# Define a funtion to compute the euclidean distance of datapoints from centroids
def compute_distances_from_centroids_with_euclidean(X,centroids):
    from scipy.spatial.distance import cdist
    eudist = cdist(X, centroids, metric='euclidean')
    labels = np.argmin(eudist, axis = 1)
    #return(eudist)
    return(labels)


# Implement the kmeans clustering algorithm
def kmeans(X,k,max_iterations, distance):
    from sklearn.metrics import silhouette_score
    import time
    start_time = time.time()
    
    minimums = np.min(X,axis = 0)
    maximums = np.max(X, axis = 0)
    nrow, ncol = X.shape
    iteration = 0
    
    centroids = np.zeros(shape = (k,ncol))
    for i in range(k):
        centroids[i,:] = np.random.uniform(minimums, maximums, size=(1,ncol))


    if distance == 'euclidean':
        while iteration <= max_iterations:
            labels = compute_distances_from_centroids_with_euclidean(X, centroids)
            new_centroids = update_centroids(X,k,labels)
            if np.array_equal(new_centroids, centroids):
                #print('{} iterations completed in total.'.format(iteration))
                break
            else:
                centroids = new_centroids
                #print(iteration)
                iteration+=1
    elif distance == 'manhattan':
        while iteration <= max_iterations:
            labels = compute_distances_from_centroids_with_manhatan(X, centroids)
            new_centroids = update_centroids(X,k,labels)
            if np.array_equal(new_centroids, centroids):
                #print('{} iterations completed in total.'.format(iteration))
                break
            else:
                centroids = new_centroids
                #print(iteration)
                iteration+=1
    
    elif distance == 'mahalanobis':
        while iteration <= max_iterations:
            labels = compute_distances_from_centroids_with_mahalanobis(X, centroids)
            new_centroids = update_centroids(X,k,labels)
            if np.array_equal(new_centroids, centroids):
                #print('{} iterations completed in total.'.format(iteration))
                break
            else:
                centroids = new_centroids
                #print(iteration)
                iteration+=1
    else:
        raise ValueError('Input Error. Distance should be Euclidean, manhattan or mahalanobis')
    


    score = silhouette_score(X, labels)
    end_time = time.time()
    run_time = end_time - start_time
    
    #print('The silhouette score is : {:.2f}'.format(score))
    #print('Clustering took {:.2f} seconds.'.format(run_time))
    return(centroids,labels, score)