## Principal Component Analysis (PCA)

PCA is an unsupervised machine learning method for dimensionality reduction. A set of multidimensional datapoints are used to compute a projection to a new lower dimensional space, the PC space, where the projected data store as much as possible of the initial information.

For this implementation two approached are followed:
* Computing the eigen vectors and eigen values of the covariance matrix.
* Applying Singular Value Decomposition (SVD) on the centered data

In any of the two approaches the data are firstly centered around their respective mean.


### <ins>Eigen value/Eigen vector approach
The covariance matrix is initially computed from the scaled data. The eigen values and the eigen vectors of the covariance matrix are computed. The eigen vectors map the data to the principal components. Each eigen vector is a mapping to a principal component, while the corresponding eigen value of the vector describes how much of the variance this eigen vector captures. As a result multiplying the scaled data with the matrix that consists of the eigen values sorted by their eigen values yields datapoints in the lower PC space where ech dimension stores less informatio from the previous dimension.


### <ins>SVD vector approach
Again the covariance matrix is computed. Singular value decomposition is a matrix factorization approach where a matrix X can be decomposed to a set of three matrices:

$$
X = U \Sigma V^T
$$
* X is the original matrix
* U is left singular matrix with dimensions mxm and the columns form an orthonormal basis
* Σ is the singular value matrix which is a diagonal matrix with the singular values on the diagonal
* V is the right singular matrix which is a nxn matrix where its columns form an orthonormal basis.

The columns of the V matrix store the directions that map the original data to the PC space. In order to get the data coordinates to the PC space the following transformation is done:

$$
Z = XV
$$



