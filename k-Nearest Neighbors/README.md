# DeepLearning4OpenCV/k-Nearest Neighbors



#

**Pseudo-Code for k-Nearest Neighbors Algorithm :**

1. Load the Dataset
2. Split the dataset into training and testing sets
3. intialise the value of k
4. train the model using training dataset
5. for getting predicted classes,iterate from 1 to total number of training data points
    * Calculate the distance between test data and each row of training data.usually we use distance function as Euclidean distance since it is most popular, alternatively we can use Manhatten distance,Hamming distance,Minkowski distance...
    
    ![Distance Functions](https://i.ibb.co/8NhH5Cy/687474703a2f2f7777772e7361656473617961642e636f6d2f696d616765732f4b4e4e5f73696d696c61726974792e706e67.png)
    
    * Sort the calculated distances in ascending order
    * Get top k rows from sorted list
    * Get the most frequent class from this list
    * return the predicted class

#

**Drawbacks :**

- It doesn't actually learn anything.
    * i.e if the algorithm makes a mistake then there is no way to learn from that mistake and "improve" itself for further classifications
    * we simply have to store the training dataset and then predictions are made on testing dataset to our training data.


- Training maybe easy but Testing is quite slow because we have to apply distance function to every training point.


- Without Any Data structures,this algorithm scales linearly with the number of data points,making it not only practically questionable to use in high dimensions(like 2D,3D,4D....),but theoretically questionable in terms of its usage.
#
**Result:**

- i used the dataset animals which consists of 3 categories in which each have 1000 sets of images.

3 categories are:
* Panda
* Dog
* Cat


####Output

![result](https://user-images.githubusercontent.com/25476729/51479092-4e35a200-1db3-11e9-9baf-e4f6fce8c37e.png)


#

Further reading [About K-Nearest Neighbors ](http://www.saedsayad.com/k_nearest_neighbors.htm)
