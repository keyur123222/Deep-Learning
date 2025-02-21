import math
import numpy as np  
from download_mnist import load
import operator  
import time
# classify using kNN  
#x_train = np.load('../x_train.npy')
#y_train = np.load('../y_train.npy')
#x_test = np.load('../x_test.npy')
#y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)  #60,000 images for training (28x28 pixels)
x_test  = x_test.reshape(10000,28,28)   #10,000 images for testing  (28x28 pixels)
x_train = x_train.astype(float)         #converts pixel value to float
x_test = x_test.astype(float)           #converts pixel value to float

# newInput: test image to classify
# dataSet: training dataset
# labels: corresponds to labels for dataSet
# k: number of nearest neighbors to consider 
def kNNClassify(newInput, dataSet, labels, k, metric): 
    result=[]
    ########################
    # Input your code here #
    ########################
    for test_sample in newInput:
        if metric == 'L1':
            distances = np.sum(np.abs(dataSet - test_sample), axis=(1, 2))  # L1 (Manhattan) Distance
        elif metric == 'L2':  # Default is L2 (Euclidean)
            distances = np.sqrt(np.sum((dataSet - test_sample) ** 2, axis=(1, 2)))  # L2 (Euclidean) Distance

        # Get indices of k nearest neighbors
        k_neighbors = np.argsort(distances)[:k]
        
        # Get corresponding labels
        k_labels = labels[k_neighbors]
        
        # Predict the most common label among the k neighbors
        unique_labels, counts = np.unique(k_labels, return_counts=True)
        result.append(unique_labels[np.argmax(counts)])
    
    ####################
    # End of your code #
    ####################
    return np.array(result)

num_train_samples = 40000
num_test_samples = 1000
k_value = 6

# Run KNN with L1 distance
start_time_L1 = time.time()
output_labels_L1 = kNNClassify(x_test[:num_test_samples], x_train[:num_train_samples], y_train[:num_train_samples], k_value, metric='L1')
accuracy_L1 = (1 - np.count_nonzero(y_test[:num_test_samples] - output_labels_L1) / len(output_labels_L1))
time_L1 = time.time() - start_time_L1

# Run KNN with L2 distance
start_time_L2 = time.time()
output_labels_L2 = kNNClassify(x_test[:num_test_samples], x_train[:num_train_samples], y_train[:num_train_samples], k_value, metric='L2')
accuracy_L2 = (1 - np.count_nonzero(y_test[:num_test_samples] - output_labels_L2) / len(output_labels_L2))
time_L2 = time.time() - start_time_L2

# Print results
print(f"--- KNN Classification Results on MNIST ---")
print(f"L1 (Manhattan) Distance -> Accuracy: {accuracy_L1:.4f}, Execution Time: {time_L1:.4f} sec")
print(f"L2 (Euclidean) Distance -> Accuracy: {accuracy_L2:.4f}, Execution Time: {time_L2:.4f} sec")