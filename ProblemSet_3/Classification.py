import numpy as np
import os
import pandas as pd

from sklearn import svm
from sklearn import model_selection

def add_bias_column(X):
    """ YOUR CODE HERE """
    length = len(X)
    bias = np.full((length, 1), 1)

    return np.hstack((bias, np.array(X)))
    """ YOUR CODE END HERE """

def cost_function(X: np.ndarray, y: np.ndarray, weight_vector: np.ndarray):
    # Machine epsilon for numpy `float64` type
    eps = np.finfo(np.float64).eps
    """ YOUR CODE HERE """
    x = np.array(X)
    y = np.array(y)
    w = np.array(weight_vector)

    p = 1 / (1 + np.exp(-x @ w))

    term1 = y * np.log(p + eps)
    term2 = (1 - y) * np.log(1 - p + eps)
    term3 = term1 + term2

    return -1 / len(x) * np.sum(term3) 
    """ YOUR CODE END HERE """

def weight_update(X: np.ndarray, y: np.ndarray, gamma: np.float64, weight_vector: np.ndarray) -> np.ndarray:
    """ YOUR CODE HERE """
    x = np.array(X)
    y = np.array(y)
    w = np.array(weight_vector)

    p = 1 / (1 + np.exp(-x @ w))
    pd = ((p - y) @ x) / len(x)

    return weight_vector - gamma * pd
    """ YOUR CODE END HERE """

def logistic_regression_classification(X: np.ndarray, weight_vector: np.ndarray, prob_threshold: np.float64=0.5):
    """ YOUR CODE HERE """
    pred = X @ weight_vector

    pred = 1 / (1 + np.exp(-pred))

    return np.where(pred < prob_threshold, 0, 1)
    
    """ YOUR CODE END HERE """

### Task 3.5: Logistic regression using stochastic gradient descent
def logistic_regression_stochastic_gradient_descent(X_train: np.ndarray, y_train: np.ndarray, max_num_iterations: int=250, threshold: np.float64=0.05, gamma: np.float64=1e-5, seed: int=43) -> np.ndarray:
    """ YOUR CODE HERE """
    weightVector = np.zeros(X_train.shape[1])
    error = cost_function(X_train, y_train, weightVector)
    numIterations = 0
    np.random.seed(seed)

    while True:
        if error <= threshold:
            break
        if numIterations >= max_num_iterations:
            break
            
        numIterations += 1
        randIdx = np.random.randint(0, len(X_train))
        sample = X_train[randIdx]
        target = y_train[randIdx]

        pred = sample @ weightVector
        pred = 1 / (1 + np.exp(-pred))

        diff = pred - target
        grad = diff * sample

        weightVector -= gamma * grad
        error = cost_function(X_train, y_train, weightVector)

    return weightVector
    """ YOUR CODE END HERE """

def linear_svm(X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3, random_state=42)
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    clf_predictions = clf.predict(X_test)
    return clf_predictions, clf.score(X_test, y_test) * 100

def gaussian_kernel_svm(X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3, random_state=42)

    gaussian_kernel_classifier = svm.SVC(kernel='rbf')
    gaussian_kernel_classifier.fit(X_train, y_train)

    gaussian_kernel_classifier_predictions = gaussian_kernel_classifier.predict(X_test)

    return gaussian_kernel_classifier_predictions, gaussian_kernel_classifier.score(X_test, y_test) * 100

