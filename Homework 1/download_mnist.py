import numpy as np
import pickle
from tensorflow.keras.datasets import mnist

filename = [
    ["training_images", "x_train"],
    ["test_images", "x_test"],
    ["training_labels", "y_train"],
    ["test_labels", "y_test"]
]

def download_mnist():
    print("Downloading MNIST dataset from TensorFlow...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape the images into 784-dimensional vectors
    x_train = x_train.reshape(-1, 28 * 28).astype(np.uint8)
    x_test = x_test.reshape(-1, 28 * 28).astype(np.uint8)

    mnist_data = {
        "training_images": x_train,
        "training_labels": y_train,
        "test_images": x_test,
        "test_labels": y_test
    }

    with open("mnist.pkl", "wb") as f:
        pickle.dump(mnist_data, f)

    print("Download and save complete.")

def load():
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

def init():
    download_mnist()

if __name__ == '__main__':
    init()
