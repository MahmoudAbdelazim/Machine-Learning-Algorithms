import numpy as np
import pandas as pd


def entropy(y):  # the entropy loss
    occ = np.bincount(y)
    ps = occ / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])  # entropy formula


class Node:
    def __init__(self, feature=None, left=None, right=None, label=None):
        self.feature = feature  # the feature that node will split by
        self.left = left  # left child node
        self.right = right  # right child node
        self.label = label  # value of the leaf node (the most common class label)

    def is_leaf(self):
        return self.label is not None


class DecisionTree:
    def __init__(self, max_depth=18):
        self.max_depth = max_depth  # maximum depth of the tree
        self.tree_depth = 0
        self.num_features = 0  # the number of features
        self.root = None  # the root node of the tree

    def train(self, X, y):
        self.num_features = X.shape[1]
        self.root = self.grow_tree(X, y)
        return self.tree_depth

    def grow_tree(self, X, y, tree_depth=0):  # recursively grow the tree
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))
        if tree_depth > self.tree_depth:
            self.tree_depth = tree_depth

        # stopping criteria
        if tree_depth >= self.max_depth or num_labels == 1:
            leaf_value = self.most_common_class(y)  # assign the leaf node a value of the most common label
            return Node(label=leaf_value)  # finish

        np.random.seed(0)
        features_indexes = np.random.choice(num_features, self.num_features, replace=False)

        # greedily select the best split according to information gain
        best_feat = self.best_feature(X, y, features_indexes)

        # grow the children subtrees that result from the split
        left_indexes, right_indexes = self.split(X[:, best_feat])
        left = self.grow_tree(X[left_indexes, :], y[left_indexes], tree_depth + 1)
        right = self.grow_tree(X[right_indexes, :], y[right_indexes], tree_depth + 1)
        return Node(best_feat, left, right)

    def most_common_class(self, y):  # returns the most common label in y (1 or 0)
        ones = sum(x for x in y if x == 1)
        zeros = sum(x for x in y if x == 0)
        if ones > zeros:
            return 1
        else:
            return 0

    def best_feature(self, X, y, feature_indexes):  # returns the best feature idx that has biggest information gain
        best_gain = -1
        bestFeatIdx = None
        for feat_idx in feature_indexes:  # loop over all features
            col = X[:, feat_idx]  # all samples of this feature
            inf_gain = self.information_gain(y, col)  # get the information gain
            if inf_gain > best_gain:  # if gain of this feature is bigger, assign it to best_gain
                bestFeatIdx = feat_idx
                best_gain = inf_gain
        return bestFeatIdx

    def information_gain(self, y, col):
        # parent entropy
        parent_entropy = entropy(y)
        # generate split
        left_indexes, right_indexes = self.split(col)
        if len(left_indexes) == 0 or len(right_indexes) == 0:
            return 0
        # compute the weighted avg. of the entropy for the children
        n = len(y)
        num_l, num_r = len(left_indexes), len(right_indexes)
        entropy_l, entropy_r = entropy(y[left_indexes]), entropy(y[right_indexes])
        avg_child_entropy = (num_l / n) * entropy_l + (num_r / n) * entropy_r
        # information gain is entropy of parent - avg entropy of children after split
        inf_gain = parent_entropy - avg_child_entropy
        return inf_gain

    def split(self, col):  # splits the data, zeros to the left and ones to the right
        left_indexes = []
        right_indexes = []
        for i in range(len(col)):
            if col[i] == 0:
                left_indexes.append(i)
            else:
                right_indexes.append(i)
        return left_indexes, right_indexes

    def predict(self, X):  # predicts the labels of a numpy array of rows
        return np.array([self.pred(x, self.root) for x in X])

    def pred(self, x, node):  # recursively traverse the tree till reaching a leaf node
        if node.is_leaf():
            return node.label

        if x[node.feature] == 0:  # if the value of the feature is 0, go left in the tree
            return self.pred(x, node.left)
        return self.pred(x, node.right)  # else go right


def adjust_data(data):
    # replace all '?' the majority of votes
    for colName, content in data.iteritems():
        col = data[colName]
        yes = sum(1 for x in col if x == 'y')
        no = sum(1 for x in col if x == 'n')
        major = 'y'
        if no > yes:
            major = 'n'
        col[:] = [major if x == '?' else x for x in col]

    # convert all 'y' to 1 and 'n' to 0
    # convert all 'democrat' to 1 and 'republican' to 0
    for colName, content in data.iteritems():
        col = data[colName]
        col[:] = [1 if x == 'y' or x == 'democrat' else 0 for x in col]
    return data


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def split_to_train_and_test(data, train_size, random_state):
    y = data.iloc[:, 0]  # target labels
    X = data.iloc[:, 1:]  # features
    X_train = X.sample(frac=train_size, random_state=random_state)
    y_train = y.sample(frac=train_size, random_state=random_state)
    X_test = X.drop(X_train.index)
    y_test = y.drop(y_train.index)

    X_train = np.array(X_train.values.tolist())
    X_test = np.array(X_test.values.tolist())
    y_train = np.array(y_train.values.tolist())
    y_test = np.array(y_test.values.tolist())

    return X_train, X_test, y_train, y_test


def init(train_size, random_state):
    # read and adjust the dataset
    print("Reading dataset ...")
    data = pd.read_csv("house-votes-84.data.txt")
    print("Adjusting the data ...")
    data = adjust_data(data)

    # ------------------------------------------------------------------
    # split to train and test
    print("Splitting data to train and test ...")
    X_train, X_test, y_train, y_test = split_to_train_and_test(data, train_size, random_state)

    # ------------------------------------------------------------------
    # create the decision tree
    print("Creating the decision tree ...")
    clf = DecisionTree(max_depth=18)
    # train the decision tree wil training features and labels
    print("Training started ...")
    tree_depth = clf.train(X_train, y_train)
    print("Training finished")

    # ------------------------------------------------------------------
    # test
    print("Testing the model ...")
    y_test_predicted = clf.predict(X_test)
    acc = accuracy(y_test, y_test_predicted)

    print("Accuracy on test data: {}".format(acc))
    print("Decision Tree Depth: {}".format(tree_depth))
    return acc, tree_depth


def statistics():
    output_file = open("model results.txt", 'w')
    train_sizes = [0.3, 0.4, 0.5, 0.6, 0.7]
    random_states = [1, 2, 3, 4, 5]
    for i in train_sizes:
        min_acc, max_acc, sum_acc = 100, 0, 0
        min_tree_depth, max_tree_depth, sum_tree_depth = 100, 0, 0
        for j in random_states:
            output_file.write("\n-----------------------------\n")
            output_file.write("Training with " + str(i * 100) + "% of data, random_state = " + str(j) + "\n")
            acc, tree_depth = init(i, j)
            if acc < min_acc:
                min_acc = acc
            if acc > max_acc:
                max_acc = acc
            sum_acc += acc
            if tree_depth < min_tree_depth:
                min_tree_depth = tree_depth
            if tree_depth > max_tree_depth:
                max_tree_depth = tree_depth
            sum_tree_depth += tree_depth
            output_file.write("Accuracy: " + str(acc) + "\n")
            output_file.write("Tree Depth: " + str(tree_depth) + "\n")
        mean_acc = sum_acc / len(random_states)
        mean_tree_depth = sum_tree_depth / len(random_states)
        output_file.write("\n-----------------------------\n")
        output_file.write("When Training with " + str(i * 100) + "% of data: \n")
        output_file.write("The Min Accuracy is: " + str(min_acc) + "\n")
        output_file.write("The Max Accuracy is: " + str(max_acc) + "\n")
        output_file.write("The Mean Accuracy is: " + str(mean_acc) + "\n")
        output_file.write("The Min Tree Depth is: " + str(min_tree_depth) + "\n")
        output_file.write("The Max Tree Depth is: " + str(max_tree_depth) + "\n")
        output_file.write("The Mean Tree Depth is: " + str(mean_tree_depth) + "\n")
        output_file.write("=================================================\n")


statistics()  # runs with different training sizes and random states and outputs the statistics to a file
# init(0.3, 1)  # runs normally (first parameter is the training size, second is the random_state)
