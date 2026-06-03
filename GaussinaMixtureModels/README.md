### <ins> Gaussian Mixture Models

**Gaussian Mixture Models (GMM)** is a supervised machine learning method used to assign the datapoints of a dataset to clusters. The core idea of the model is that the dataset contains datapoints that are sampled from different distributions. Each cluster is seen as a **different** Gaussian distributions. The goal is to find and cluster together datapoints that are sampled from the same distribution.

Initially we assume that the datapoints are sampled from k different Gaussians. Instead of strictly assigning a datapoint to a cluster(Gaussian) k probabilities are assigned to the datapoint, each one for every Gaussian. So the model is formed as :


$$
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)
$$

Where:
- \(p(x)\) is the probability density function of x 
- \(K\): total number of clusters  
- \(\pi_k\): mixing weight (how probable is that \(x\) comes from \(k\))  
- \(\mu_k\): mean of cluster \(k\)  
- \(\Sigma_k\): covariance (shape of cluster \(k\))  
- \(\mathcal{N}(x \mid \mu_k, \Sigma_k)\): Gaussian distribution for cluster \(k\)


**<ins>Fitting the model**
In order to fit the model the expectation maximization approach is followd in order to update the mean and the variance of the Gaussians as well as the \(π_k\) 