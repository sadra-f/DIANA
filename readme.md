## An implementation of a Top-Down (Divisive) Hierarchical Clustering Algorithm


### Algorithm Steps:
- Compute the distance matrix for the dataset.
- Put all elements into one cluster.
- repeat:
    - Select the cluster which has the largest average dissimilarity within it.
    - Find the element in selected cluster which is most dissimilar to the rest of the data and move it to a new cluster.
    - Find & move the elements from the selected cluster that are similar enough to the new cluster that they can be moved to the new cluster.
    - save the state of all clusters as history.

### Dataset:
Used dataset is a [wine dataset](https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering) from Kaggle.