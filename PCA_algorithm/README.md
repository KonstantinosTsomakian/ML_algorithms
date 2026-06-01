## Principal Component Analysis (PCA)

PCA is an unsupervised machine learning method for dimensionality reduction. A set of multidimensional datapoints are used to compute a projection to a new lower dimensional space, the PC space, where the projected data store as much as possible of the initial information.

For this implementation two approached are followed:
* Computing the eigen vectors and eigen values of the covariance matrix.
* Applying Singular Value Decomposition (SVD) on the centered data

In any of the two approaches the data are firstly centered around their respective mean.

### <ins>Eigen value/Eigen vector approach
The covariance matrix is initially computed from the scaled data. The eigen values and the eigen vectors of the covariance matrix are computed. The eigen vectors map the data to the principal components. Each eigen vectors is a mapping to a principal component, while the corresponding eigen value of the vector describes how much of the variance this eigen vector captures. As a result multiplying the scaled data with the matrix that consists of the eigen values sorted by their eigen values yields datapoints in the lower PC space where ech dimension stores less informatio from the previous dimension.


## Features

- Easy to read
- Easy to write
- Plain text format

### Example Code

```python
print("Hello, Markdown!")
```

**Bold text**

*Italic text*


