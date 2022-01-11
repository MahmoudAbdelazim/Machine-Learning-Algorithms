import numpy as np
import pandas as pd


def compute_cost(W, b, X, Y):
    N = X.shape[0]
    fx = hx(X, W, b)
    dist = 1 - Y * fx
    dist[dist < 0] = 0  # max(0, y * f(x))
    cost = (lambdaa / 2) * np.dot(W, W) + (1 / N) * (np.sum(dist))  # cost formula
    return cost


def train_with_gradient_descent(X, y):
    W = np.zeros(X.shape[1])  # weights
    b = 0  # intercept
    for epoch in range(epochs):
        for i, xi in enumerate(X):
            if y[i] * hx(xi, W, b) >= 1:  # if the point is correctly classified
                W -= learning_rate * (2 * lambdaa * W)
            else:  # if not correctly classified
                W += learning_rate * (np.dot(xi, y[i]) - 2 * lambdaa * W)
                b += learning_rate * y[i]

        cost = compute_cost(W, b, X, y)  # calculate cost at each GD iteration
        print("Epoch : {},  Cost : {}".format(epoch + 1, cost))

    return W, b


def hx(x, W, b):  # Hypothesis function
    return np.dot(x, W) + b


def predict(batch, W, b):  # predicts the labels of a numpy array
    predicted = np.array([])
    for i in range(batch.shape[0]):  # loop over the elements of the numpy array
        p = np.sign(hx(batch.to_numpy()[i], W, b))  # get the predicted class
        predicted = np.append(predicted, p)  # add it to the predicted list
    return predicted


def normalize(col):
    # Z-score normalization
    col = (col - col.mean()) / col.std()
    return col


def split_to_train_and_test(data, train_size, random_state):
    Y = data.loc[:, 'target']  # target labels
    X = data.iloc[:, 0:-1]  # features
    for col in X:
        X[col] = normalize(X[col])
    X_train = X.sample(frac=train_size, random_state=random_state)
    Y_train = Y.sample(frac=train_size, random_state=random_state)
    X_test = X.drop(X_train.index)
    Y_test = Y.drop(Y_train.index)
    return X_train, X_test, Y_train, Y_test


def accuracy(pred, Y_test):
    correct = 0
    for i in range(len(Y_test.values)):
        if pred[i] == Y_test.values[i]:
            correct += 1
    acc = correct / len(Y_test.values)
    return acc


def init(train_size, random_state):
    # read and adjust the dataset
    print("Reading dataset ...")
    data = pd.read_csv('heart.csv')
    map_classes = {1: 1, 0: -1}
    data['target'] = data['target'].map(map_classes)
    # ------------------------------------------------------------------
    # split to train and test
    print("Splitting data to train and test ...")
    X_train, X_test, Y_train, Y_test = split_to_train_and_test(data, train_size, random_state)

    # ------------------------------------------------------------------
    # Training
    print("Started training ...")
    W, b = train_with_gradient_descent(X_train.to_numpy(), Y_train.to_numpy())
    print("Finished training ...")

    print("weights (W) are: {}".format(W))
    print("Intercept (b) is: {}".format(b))

    # ------------------------------------------------------------------
    # Testing
    print("Testing the model ...")
    Y_test_predicted = predict(X_test, W, b)
    acc = accuracy(Y_test_predicted, Y_test)
    print("Accuracy on test data: {}".format(acc))
    return acc


def statistics():
    output_file = open("model results.txt", 'w')
    train_sizes = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    random_states = [1, 2, 3, 4, 5]
    for i in train_sizes:
        min_acc, max_acc, sum_acc = 100, 0, 0
        for j in random_states:
            output_file.write("\n-----------------------------\n")
            output_file.write("Training with " + str(i * 100) + "% of data, random_state = " + str(j) + "\n")
            acc = init(i, j)
            if acc < min_acc:
                min_acc = acc
            if acc > max_acc:
                max_acc = acc
            sum_acc += acc
            output_file.write("Accuracy: " + str(acc) + "\n")
        mean_acc = sum_acc / len(random_states)
        output_file.write("\n-----------------------------\n")
        output_file.write("When Training with " + str(i * 100) + "% of data: \n")
        output_file.write("The Min Accuracy is: " + str(min_acc) + "\n")
        output_file.write("The Max Accuracy is: " + str(max_acc) + "\n")
        output_file.write("The Mean Accuracy is: " + str(mean_acc) + "\n")
        output_file.write("=================================================\n")


learning_rate = 0.001
epochs = 50
lambdaa = 1 / epochs

statistics()  # runs with different training sizes and random states and outputs the statistics to a file
# init(0.3, 1)  # runs normally (first parameter is the training size, second is the random_state)
