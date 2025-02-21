import numpy as np
import pickle
from download_mnist import load

x_train, y_train, x_test, y_test = load()

x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding function
def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

# One-hot encode the labels
num_classes = 10
y_train_onehot = one_hot_encode(y_train, num_classes)
y_test_onehot = one_hot_encode(y_test, num_classes)

num_features = 28 * 28  # 784 for 28x28 pixel images

np.random.seed(42) # Initialize weights and bias
W = np.random.randn(num_features, num_classes) * 0.01
b = np.zeros((1, num_classes))

def cross_entropy_loss(y_true, y_pred): # Cross-entropy loss function
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
    return loss

def softmax(logits): # Softmax function
    exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def train_linear_classifier(x_train, y_train_onehot, learning_rate=0.1, epochs=100): # Training function
    global W, b
    for epoch in range(epochs):
        # Compute logits and predictions
        logits = np.dot(x_train, W) + b
        y_pred = softmax(logits)

        # Compute loss
        loss = cross_entropy_loss(y_train_onehot, y_pred)

        # Gradient computation
        m = x_train.shape[0]
        dW = np.dot(x_train.T, (y_pred - y_train_onehot)) / m
        db = np.sum(y_pred - y_train_onehot, axis=0, keepdims=True) / m

        # Update weights and bias
        W -= learning_rate * dW
        b -= learning_rate * db

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")


def compute_accuracy(x, y_true_onehot): # Evaluate accuracy
    logits = np.dot(x, W) + b
    y_pred = np.argmax(softmax(logits), axis=1)
    y_true = np.argmax(y_true_onehot, axis=1)
    return np.mean(y_pred == y_true)

# Train the linear classifier
train_linear_classifier(x_train, y_train_onehot)

# Evaluate on the testing set
accuracy = compute_accuracy(x_test, y_test_onehot)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Random Search for Weight Optimization
def random_search(x_train, y_train_onehot, num_searches=50):
    best_accuracy = 0
    best_W = None
    best_b = None

    # poop code (not really sure about this part)
    for i in range(num_searches):
        # Generate random weights and biases
        random_W = np.random.randn(num_features, num_classes) * 0.01
        random_b = np.random.randn(1, num_classes) * 0.01

        # Compute accuracy for the current random parameters
        logits = np.dot(x_test, random_W) + random_b
        y_pred = np.argmax(softmax(logits), axis=1)
        y_true = np.argmax(y_test_onehot, axis=1)
        accuracy = np.mean(y_pred == y_true)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_W = random_W
            best_b = random_b

    print(f"Best accuracy after random search: {best_accuracy * 100:.2f}%")
    return best_W, best_b

# Perform random search
best_W, best_b = random_search(x_train, y_train_onehot)