## <ins> K means clustering

**K means clustering is a simple supervised machine learning algorithm used to assign datapoints of a dataset to distinct clusters.**

* The user defines a distance metric and the number of k desired clusters

**1)** The algorithm randomly initializes k datapoints as starting clusters and subsequently computes all the pairwise distances between the datapoints and the defined points. 

**2)** Each datapoint is assigned to the nearest cluster based on the distance metric defined. 

**3)** Clusters of data points are formed and for each cluster the mean datapoint/centroid is computed.

**4)** The distances of all datapoints to the new centroids are reeveluated.

The process of computing new clusters and evaluating the distances from the centroids is repeated until no changes in the clusters appear.

<p align="center">
  <img src="images/Screenshot%202026-06-03%20181029.png" width="300">
</p>