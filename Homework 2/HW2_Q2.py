import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# load mnist data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# preprocess images and labels
train_images = train_images.reshape(-1, 28 * 28) / 255.0
test_images = test_images.reshape(-1, 28 * 28) / 255.0
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# initialize the model parameters
np.random.seed(42)
weights = np.random.randn(28 * 28, 10) * 0.01  # weights (784, 10)
biases = np.zeros((1, 10))  # biases vector (1, 10)
learning_rate = 0.1  # the rate of learning
num_epochs = 100
batch_size = 128

# softmax function 
def softmax_function(z):
    exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# the training loop
for epoch in range(num_epochs):
    for i in range(0, train_images.shape[0], batch_size):
        batch_images = train_images[i:i + batch_size]
        batch_labels = train_labels[i:i + batch_size]
        
        # forward pass
        logits = np.dot(batch_images, weights) + biases
        predictions = softmax_function(logits)

        # compute loss using cross-entropy
        loss = -np.mean(np.sum(batch_labels * np.log(predictions + 1e-9), axis=1))

        # gradients computing
        grad_weights = np.dot(batch_images.T, (predictions - batch_labels)) / batch_size
        grad_biases = np.sum(predictions - batch_labels, axis=0, keepdims=True) / batch_size

        # weights and biases update
        weights -= learning_rate * grad_weights
        biases -= learning_rate * grad_biases

    # print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch + 1}: loss = {loss:.4f}")

# test the model
test_logits = np.dot(test_images, weights) + biases
test_predictions = np.argmax(softmax_function(test_logits), axis=1)
test_true_labels = np.argmax(test_labels, axis=1)

accuracy = np.mean(test_predictions == test_true_labels) * 100
print(f"test accuracy: {accuracy:.2f}%")
