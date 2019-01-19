# DeepLearning4OpenCV/k-Nearest Neighbors

i used the dataset animals which consists of 3 categories in which each have 1000 sets of images.

3 categories are:
* Panda
* Dog
* Cat

**Pseudo-Code for k-Nearest Neighbors Algorithm :**

1. Load the Dataset
2. Split the dataset into training and testing sets
3. intialise the value of k
4. train the model using training dataset
5. for getting predicted classes,iterate from 1 to total number of training data points
  * Calculate the distance between test data and each row of training data.usually we use distance function
  as Euclidean distance since it is most popular, alternatively we can use Manhatten distance,Hamming distance,Minkowski distance...
  ![Distance Functions](http://www.saedsayad.com/images/KNN_similarity.png)
  
  * Sort the calculated distances in ascending order
  * Get top k rows from sorted list
  * Get the most frequent class from this list
  * return the predicted class

Further reading [About K-Nearest Neighbors ](http://www.saedsayad.com/k_nearest_neighbors.htm)
