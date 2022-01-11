import numpy as np
import pandas as pd


# Hypothesis Function
def predict(X, w):
    return np.matmul(X, w)


# Mean Squared Error
def compute_cost(w, X, Y):
    m = Y.shape[0]
    Y_pred = predict(X, w)
    cost = (Y - Y_pred) ** 2
    return np.sum(cost) / (2 * m)


def train_with_gradient_descent(epochs, lr, X, Y):
    w = np.zeros(X.shape[1])
    for epoch in range(epochs):
        h = predict(X, w) # get current prediction
        gradient = np.dot(X.T, (h - Y)) / Y.shape[0] # gradient steps
        w = w - lr * gradient # apply gradient
        print("Epoch: {}, Cost is: {}".format(epoch + 1, compute_cost(w, X, Y)))
    return w


# Z-score normalization
def normalize(col):
    col = (col - col.mean()) / col.std()
    return col


def split_data(data):
    Y = data.loc[:, 'price']  # target labels
    X = data.loc[:, ['grade', 'bathrooms', 'lat', 'sqft_living', 'view']]  # features
    for col in X.columns:
        X[col] = normalize(X[col])
    Y = normalize(Y)
    X_train = X.sample(frac=1.0, random_state=1)
    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train.to_numpy()), axis=1)
    Y_train = Y.sample(frac=1.0, random_state=1)
    return X_train, Y_train.to_numpy()


def init():
    # read the dataset
    print("Reading dataset ...")
    data = pd.read_csv("house_data.csv")
    # ------------------------------------------------------------------
    # split data
    print("Splitting data to features and labels ...")
    X, Y = split_data(data)
    # ------------------------------------------------------------------
    # Training
    print("Started training ...")
    w = train_with_gradient_descent(100, 0.02, X, Y)
    print("Finished training ...")
    print("weights (W) are: {}".format(w))


init()
