# Compute the principal components using the eigen
# vectors and values of the covariance matrix
def eig_PCA(X, keep_dims):

    import numpy as np

    #Center the data
    mean_X = np.mean(X, axis = 0)

    centered_X = X - mean_X



    #Compute the covariance matrix of the standardized matrix X


    number_of_features = X.shape[1]
    feature_means = np.mean(centered_X, axis = 0)
    cov_step_1 = centered_X - feature_means
    n = centered_X.shape[0]
    cov_X = np.zeros((20,20))


    for feature1 in range(number_of_features):
        for feature2 in range(number_of_features):
            #print(feature1)
            #print(feature2)
            cov_X[feature1,feature2] = (sum((cov_step_1[:,feature1]) * (cov_step_1[:,feature2]))) / (n-1)

        

    #Compute the eigen values and eigen vectors of the covariance matrix

    eigenvalues_cov_X, eigenvectors_cov_X = np.linalg.eig(cov_X)

    #Sort the eigenvalues and eigenvectors in descending eigenvalue order

    indeces_for_sorting = np.argsort(eigenvalues_cov_X)[::-1]
    sorted_eigenvalues_cov_X = eigenvalues_cov_X[indeces_for_sorting]
    sorted_eigenvectors_cov_X = eigenvectors_cov_X[indeces_for_sorting]

    #Project the standardized X matrix to the eigen vectors

    projected_data = np.dot(centered_X, sorted_eigenvectors_cov_X)

    #Compute the loadings of the features

    loadings = sorted_eigenvectors_cov_X * sorted_eigenvalues_cov_X

    #Compute the explained variance of each principal component
    explained_variance = sorted_eigenvalues_cov_X / sum(sorted_eigenvalues_cov_X)
    explained_variance[0] + explained_variance[1]


    return(projected_data[:,:keep_dims], loadings[:,:keep_dims], explained_variance[:keep_dims])


# Compute the principal components 
# using singulr value decomposition of the matrix

def svd_PCA(X, keep_dims):
    import numpy as np
    from matplotlib import pyplot as plt
    #SVD approach

    #Center the data

    mean_X = np.mean(X, axis = 0)
    std_X = np.std(X, axis = 0)
    centered_X = X - mean_X


    #SVD
    U, Sigma, VT = np.linalg.svd(centered_X)
    V = VT.T

   

    #Project the data
    svd_projected_data = np.dot(centered_X, V)

    #Compute the loadings
    n = centered_X.shape[0]
    svd_loadings = (V * Sigma) / (n-1)
    

    #compute the explained variance
    
    explained_variance = (Sigma * Sigma) / sum((Sigma * Sigma))

    
    
    return(svd_projected_data[:,:keep_dims], svd_loadings[:,:keep_dims], explained_variance[:keep_dims])



# Define a function to run the algorithm
def PCA(data, keep_dims, method):
    if method == 'eig':
        result = eig_PCA(X=data, keep_dims = keep_dims)
        return(result)
        
    elif method == 'svd':
        result = svd_PCA(X=data, keep_dims = keep_dims)
        return(result)

    else:
        raise ValueError("Input Error. Method should be 'eig' or 'svd'")
    
    