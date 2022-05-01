"""
Programmer: Riley Sikes
Class: CPSC 322-01, Spring 2021
Programming Assignment #6
3/30/22
Notes: if I dont use range(len()) my program doesn't work.
Linter tries to get me to use enumerate but it breaks my code
Description: This program contains a series of testing classes
"""

import operator
import copy
from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
import mysklearn.myevaluation as myevaluation

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        regression = MySimpleLinearRegressor()
        self.regressor = regression
        regression.fit(X_train, y_train)
        return regression.slope, regression.intercept

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = self.regressor.predict(X_test)
        return self.discretizer(predictions)

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        for j in range(len(X_test)):
            new_distances = []
            for i in range(len(self.X_train)):
                new_distances.append([myutils.compute_euclidean_distance(self.X_train[i], X_test[j]), i, self.y_train[i]])
            distances.append(new_distances)

        # sort each inner array
        for j in range(len(distances)):
            # print("j", distances[j])
            for i in range(len(distances[j])):
                distances[j].sort()

        k_distances = []
        k_indexes = []

        for j in range(len(X_test)):
            new_distances = []
            new_predictions = []
            for i in range(self.n_neighbors):
                new_distances.append(distances[j][i][0])
                new_predictions.append(distances[j][i][1])
            k_distances.append(new_distances)
            k_indexes.append(new_predictions)

        return k_distances, k_indexes

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        distances, indexes = self.kneighbors(X_test)
        # for the linter :{
        distances = distances.append([])

        classes = [[] for _ in enumerate(X_test)]

        # build up classes list
        for k in range(len(X_test)):
            for i in range(len(self.y_train)):
                if [self.y_train[i]] not in classes[k]:
                    classes[k].append([self.y_train[i]])
        # add 0 to each
        for k in range(len(classes)):
            for i in range(len(classes[k])):
                for _ in range(len(classes[k][i])):
                    classes[k][i].append(0)

        predicted = []

        # get the class counts
        for k in range(len(indexes)):
            for j in range(len(classes[k])):
                for i in range(len(indexes[k])):
                    if classes[k][j][0] == self.y_train[indexes[k][i]]:
                        classes[k][j][1] += 1

        # sort each of them
        for k in range(len(classes)):
            for j in range(len(classes[k])):
                classes[k].sort(key=operator.itemgetter(-1), reverse=True)

        # now get the predictions
        for i in range(len(X_test)):
            predicted.append(classes[i][0][0])

        return predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        # print(X_train)
        classes = []
        # build up classes list
        for k in range(len(X_train)):
            for i in range(len(y_train)):
                if [y_train[i]] not in classes:
                    classes.append([y_train[i]])

        # add 0 to each
        for k in range(len(classes)):
            for _ in range(len(classes[k])):
                classes[k].append(0)

        # now go through and add the occurences of each
        for i in range(len(y_train)):
            for j in range(len(classes)):
                if classes[j][0] == y_train[i]:
                    classes[j][1] += 1

        # now sort
        for _ in enumerate(classes):
            classes.sort(key=operator.itemgetter(-1), reverse=True)

        # grab the most common label
        self.most_common_label = classes[0][0]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self.most_common_label for _ in X_test]


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """

        # go through and create a dictionary with class : [0, size of y_train]
        priors = {}
        for i in range(len(y_train)):
            if y_train[i] not in priors:
                priors[y_train[i]] = [1, len(y_train)]
            else:
                # adding 1 every time the class is already in the dictionary
                val = priors[y_train[i]]
                val[0] += 1
                priors[y_train[i]] = val

        self.priors = priors
        
        # get the keys
        keys = list(priors.keys())

        # for len of keys
        posteriors = {}
        for i in range(len(keys)):
        #   count = 0
            curr_att = 0
        #   for len of X_train[0] (number of attributes)
            class_posterior = {}
            for j in range(len(X_train[0])):
        #       for len of X_train (number of rows)
                inner_dict = {}
                for x in range(len(X_train)):
        #           get the unique attributes and get counts of them
        #           use a dictionary with attribute_name : [0, (size from priors at current_prior[0]))]
        #           if not in dict, and doesn't match class , add with 0
        #           if not in dict and matches class, add with 1
                    if X_train[x][j] not in inner_dict:
                        if y_train[x] != keys[i]:
                            inner_dict[X_train[x][j]] = [0, priors.get(keys[i])[0]]
                        else:
                            inner_dict[X_train[x][j]] = [1, priors.get(keys[i])[0]]
                    else:
                        # adding 1 every time the att is already in the dictionary and matches class
                        if y_train[x] == keys[i]:
                            val = inner_dict[X_train[x][j]]
                            val[0] += 1
                            inner_dict[X_train[x][j]] = val
        #       create a dictionary entry with count :  dictionary from previous for loop
                class_posterior[curr_att] = inner_dict
        #       ++count
                curr_att += 1
        #   now add to the main dictionary at the current prior with dictionary from previous list
            posteriors[keys[i]] = class_posterior

        self.posteriors = posteriors

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # print(self.posteriors)
        # make a list of preds
        preds = []
        # for len(X_test)
        for i in range(len(X_test)):
        #   make a list for vals to compare for each class, intitializaing with the posteriors
            vals = [[list(self.priors.keys())[i], self.priors.get(list(self.priors.keys())[i])[0] / self.priors.get(list(self.priors.keys())[i])[1]] for i in range(len(self.priors))]
            vals.sort()
        #   need to calc values for each class
        #   for len(priors)
            for j in range(len(vals)):
        #       need to compute each att
        #       for len(X_test[i])
                for x in range(len(X_test[i])):
        #           list_val *= posteriors at curr prior, at curr attribute, [0] / [1]
                    # print(self.posteriors.get(vals[j][0]).get(x), X_test[i][x])
                    if self.posteriors.get(vals[j][0]).get(x).get(X_test[i][x]) is None:
                        vals[j][1] * 0
                    else:
                        num = self.posteriors.get(vals[j][0]).get(x).get(X_test[i][x])[0]
                        den = self.posteriors.get(vals[j][0]).get(x).get(X_test[i][x])[1]
                        # print(num, den)
                        vals[j][1] *=  num / den 
        #   then append largest value key to list of preds
            vals.sort(key=operator.itemgetter(-1), reverse=True)
            preds.append(vals[0][0])

        return preds


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.tree = myutils.fit_starter_code(X_train, y_train)
        # print("IN FIT", self.tree)

    def predict(self, X_test, class_labels):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
            class_labels (list of class_labels): for handling unseen possible attributes

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        # predict for each instance in X_test
        # print("Tree:", self.tree)
        predictions = []
        for i in range(len(X_test)):
            pred = myutils.traverse_tree(self.tree, X_test[i], class_labels)
            if pred is None:
                print("NO PREDICTION")
            predictions.append(pred)
        return predictions

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        myutils.print_decision_rules_recursive(self.tree, [], attribute_names, class_name)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this

class MyRandomForestClassifier:
    """Represents a decision tree classifier.

    Attributes:
        forest(list MyDecisionTreeClassifiers): The list of decision trees.
        best_trees (list MyDecisionTreeClassifiers): The list of the M best decision trees.
        N(int): The number of trees to generate 
        M(int): The number of top trees to grab
        F(int): The number of attributes to use for each tree
        data(list of list): The data to predict over.
        class_label(int): the index of the class_label to predict
        accuracy(double): the accuracy of the M best trees
        class_labels(list): helpful for keeping track of possible classes
    Notes:
        Split into a 1/3 for test and 2/3 for classifier stratified tests
        For N
            Use bootstrapping to create a 63% training and 36% validation set
            Reduced the test and and training down to just F attributes
            Create a classifier with the F attribute set
            Determine accuracy with the validation set
        Pick the best M classifiers
        Use the test set from the first step to determine the accuracy with simple maj voting 
    """
    def __init__(self, N, M, F):
        """Initializer for MyRandomForestClassifier
        """
        self.forest = None
        self.best_trees = None
        self.N = N
        self.M = M
        self.F = F
        self.data = None
        self.class_index = None
        self.accuracy = None
        self.class_labels = None
    
    def fit(self, data, class_index, class_labels):
        """Takes in the data and generates all the trees
        Args:
            data (list of lists): The data to predict over
            class_index (int): the index of the class label
        """
        self.class_index = class_index
        self.data = data
        self.class_labels = class_labels

        # split into 1/3 test 2/3 classifier set
        # need to ensure that the class split is even in both
        y = [data[i][class_index] for i in range(len(data))]
        X = []
        for i in range(len(data)):
            temp = data[i]
            temp.pop(class_index)
            X.append(temp)
        # print("X", X)
        # print("y", y)
        # print("data",self.data)
        test, remainder = myutils.random_forest_stratified_splitter(X, y)
        # print("test",test, "remainder",remainder)

        # now grab the actual rows
        # print("data", data)
        X_test = [X[val] for val in test]
        y_test = [y[val] for val in test]
        # print("X_test",X_test)
        # print("y_test", y_test)

        X_remainder = [X[val] for val in remainder]
        y_remainder = [y[val] for val in remainder]
        # print("X_remainder",X_remainder)
        # print("y_remainder",y_remainder)

        # now build the classifiers
        # sample is training while out of bad is validation

        # init the forest
        forest = []

        # put all the col indexes into a list for selecting random F
        attributes = [i for i in range(len(X[0]))]

        for i in range(self.N):
            X_sample, X_out_of_bag, y_sample, y_out_of_bag = myevaluation.bootstrap_sample(X_remainder, y_remainder)
            # print(len(X_sample), len(y_sample))
            # print(len(X_out_of_bag), len(y_out_of_bag))
            chosen_attributes = myutils.random_attribute_selection(attributes, self.F)
            # sorting so we can keep track of attributes better
            chosen_attributes.sort()
            # print(chosen_attributes)
            
            # now we need to reduce the X_sample to just those attributes
            X_reduced_sample = []
            # print(X_sample)
            for i in range(len(X_sample)):
                samp = []
                for val in chosen_attributes:   
                    samp.append(X_sample[i][val])
                X_reduced_sample.append(samp)
            # also reduce X_out_of_bag
            X_reduced_out_of_bag = []
            for i in range(len(X_out_of_bag)):
                out_of_bag = []
                for val in chosen_attributes:   
                    out_of_bag.append(X_out_of_bag[i][val])
                X_reduced_out_of_bag.append(out_of_bag)

            # now we have a reduced cols and y remainder to pass into a decision tree classifier
            tree = MyDecisionTreeClassifier()
            tree.fit(X_reduced_sample, y_sample)
            # print("Tree in fit",tree.tree)
            
            y_pred = tree.predict(X_reduced_out_of_bag, self.class_labels)
            correct = 0
            # now calculate the accuracy
            for i in range(len(y_pred)):
                if y_pred[i] == y_out_of_bag[i]:
                    correct += 1
            accuracy = round(correct / len(y_pred), 3)
            forest.append([tree, chosen_attributes, accuracy])
        # now assign to actual forest var
        self.forest = forest

        # find the best M now
        forest.sort(reverse=True,key=operator.itemgetter(-1))

        best_trees = forest[:self.M]
        self.best_trees = best_trees

        # now find the accuracy of the ensemble
        voted_y_pred = self.predict(X_test)
        
        # find ensemble accuracy now
        correct = 0
        # now calculate the accuracy
        for i in range(len(voted_y_pred)):
            if voted_y_pred[i] == y_test[i]:
                correct += 1
        self.accuracy = round(correct/len(voted_y_pred), 3)


    def predict(self, X_test):
        """Looks at the best trees and makes a prediction based off of 
           majority voting
        Returns:
            the prediction
        """

        y_pred = []

        for i in range(len(self.best_trees)):
            # print(X_test)
            # print("chosen attributes", self.best_trees[i][1])
            # customize a copy of Xtest for this run
            X_test_reduced = copy.deepcopy(X_test)
            for j in range(len(X_test)):
                temp = []
                for val in self.best_trees[i][1]:
                    temp.append(X_test[j][val])
                X_test_reduced[j] = temp
            # print("Tree",self.best_trees[i][0].tree)
            y_pred.append(self.best_trees[i][0].predict(X_test_reduced, self.class_labels))
        # print(y_pred, y_test)
        # print(y_pred)
        voted_y_pred = myutils.majority_vote(y_pred)
        
        return voted_y_pred