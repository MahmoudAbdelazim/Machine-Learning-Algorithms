import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hypothesis Function
def predict(X, w):
    return sigmoid(np.dot(X, w))


# cost function
def compute_cost(w, X, Y):
    m = len(Y)
    pred = predict(X, w)
    j = Y * np.log(pred) + (1 - Y) * np.log(1 - pred)
    return -(np.sum(j) / m)


def train_with_gradient_descent(epochs, lr, X, Y):
    w = np.zeros(X.shape[1])
    for epoch in range(epochs):
        h = predict(X, w)  # get current prediction
        gradient = np.dot(X.T, (h - Y)) / Y.shape[0]  # gradient steps
        w = w - lr * gradient  # apply gradient
        print("Epoch: {}, Cost is: {}".format(epoch + 1, compute_cost(w, X, Y)))
    return w


# Z-score normalization
def normalize(col):
    col = (col - col.mean()) / col.std()
    return col


def split_to_train_and_test(data, train_size):
    Y = data.loc[:, 'target']  # target labels
    X = data.iloc[:, 0:-1]  # features
    for col in X.columns:
        X[col] = normalize(X[col])
    X_train = X.sample(frac=train_size, random_state=1)
    Y_train = Y.sample(frac=train_size, random_state=1)
    X_test = X.drop(X_train.index)
    Y_test = Y.drop(Y_train.index)
    return X_train, X_test, Y_train, Y_test


def accuracy(pred, Y_test):
    correct = 0
    for i in range(len(Y_test.values)):
        p = 0
        if pred[i] >= 0.5:
            p = 1
        if p  == Y_test.values[i]:
            correct += 1
    acc = correct / len(Y_test.values)
    return acc


def init():
    # read the dataset
    print("Reading dataset ...")
    data = pd.read_csv("heart.csv")
    # ------------------------------------------------------------------
    # split to train and test
    print("Splitting data to train and test ...")
    X_train, X_test, Y_train, Y_test = split_to_train_and_test(data, 0.8)
    # ------------------------------------------------------------------
    # Training
    print("Started training ...")
    W  = train_with_gradient_descent(100, 0.1, X_train.to_numpy(), Y_train.to_numpy())
    print("Finished training ...")
    print("weights (W) are: {}".format(W))

    # ------------------------------------------------------------------
    # Testing
    print("Testing the model ...")
    Y_test_predicted = predict(X_test, W)
    acc = accuracy(Y_test_predicted, Y_test)
    print("Accuracy on test data: {}".format(acc))


init()
