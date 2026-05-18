import os
import numpy as np

def load_data(filepath):
    *X, y = np.genfromtxt(
        filepath,
        delimiter=',',
        skip_header=True,
        unpack=True,
    ) # default dtype: float
    X = np.array(X, dtype=float).T # cast features to int type
    return X, y
data_filepath = 'housing_data.csv'
X, y = load_data(data_filepath)

def mean_squared_error(y_true, y_pred):
    """ YOUR CODE HERE """
    length = len(y_pred)
    true = np.array(y_true)
    pred = np.array(y_pred)

    return np.sum(np.square(pred - true)) / (2 * length)
    """ YOUR CODE END HERE """

def mean_absolute_error(y_true, y_pred):
    """ YOUR CODE HERE """
    length = len(y_pred)
    true = np.array(y_true)
    pred = np.array(y_pred)

    return np.sum(np.abs(pred - true)) / (length)
    """ YOUR CODE END HERE """

def add_bias_column(X):
    """ YOUR CODE HERE """
    length = len(X)
    bias = np.full((length, 1), 1)
    return np.hstack((bias, np.array(X)))
    """ YOUR CODE END HERE """

def get_bias_and_weight(X, y, include_bias = True):  # Normal Eqn
    """ YOUR CODE HERE """
    newX = X
    newXTranspose = np.transpose(newX)

    if include_bias:
        newX = add_bias_column(X)
        newXTranspose = np.transpose(newX)

    bias = 0
    weights = np.linalg.inv(newXTranspose @ newX) @ newXTranspose @ y

    if include_bias:
        bias = weights[0]
        weights = weights[1:]

    return bias, weights
    """ YOUR CODE END HERE """

def get_prediction_linear_regression(X, y, include_bias = True):
    """ YOUR CODE HERE """
    bias, weights = get_bias_and_weight(X, y, include_bias)

    return np.array(X) @ weights + bias
    """ YOUR CODE END HERE """

def create_polynomial_matrix(X, power = 2):
    """ YOUR CODE HERE """
    X = np.array(X)
    X = np.tile(X, (1, power))
    X = np.cumprod(X, axis = 1)
    return X
    """ YOUR CODE END HERE """

def get_prediction_poly_regression(X, y, power = 2, include_bias = True):
    """ YOUR CODE HERE """
    X = np.array(X)
    X = create_polynomial_matrix(X, power)

    return get_prediction_linear_regression(X, y, include_bias)
    """ YOUR CODE END HERE """

def gradient_descent_multi_variable(X, y, lr=1e-5, number_of_epochs=250):
    bias = 0.0
    weights = np.full((X.shape[1], ), 0).astype(float)
    loss = []

    n = X.shape[0]
    pred = X @ weights + bias  
    for _ in range(number_of_epochs):
        pred = X @ weights + bias    
        diff = pred - y
        
        grad_w = (1/n) * (X.T @ diff) 
        grad_b = (1/n) * np.sum(diff) 
        
        weights -= lr * grad_w
        bias -= lr * grad_b
        
        mse = mean_squared_error(y, pred)
        loss.append(mse)
    
    return bias, weights, loss

def feature_scaling(X):
    """ YOUR CODE HERE """
    X = np.array(X)
    mean = np.mean(X, axis = 0)
    sd = np.std(X, axis = 0)

    return (X - mean) / sd
    """ YOUR CODE END HERE """

### Task 2.8: Finding the number of epochs for gradient descent to converge
def find_number_of_epochs(X, y, lr, delta_loss):
    bias = 0
    weights = np.full((X.shape[1], ), 0).astype(float)
    num_of_epochs = 0
    previous_loss = 1e14
    current_loss = -1e14
    n = X.shape[0]
    pred = X @ weights + bias
    current_loss = mean_squared_error(y, pred)
    while abs(previous_loss - current_loss) >= delta_loss:
        """ YOUR CODE HERE """
        num_of_epochs += 1       
        diff = pred - y
        grad_w = (1/n) * (X.T @ diff) 
        grad_b = (1/n) * np.sum(diff) 

        weights -= lr * grad_w
        bias -= lr * grad_b
    
        pred = X @ weights + bias      
        previous_loss = current_loss
        current_loss = mean_squared_error(y, pred)
        """ YOUR CODE END HERE """
    return bias, weights, num_of_epochs, current_loss


def compare_quality(X, y, lr=1e-5, include_bias=True, delta_loss=1e-7, scaling = False):
    
    print("===== Comparing Normal Equation vs Gradient Descent =====")
    
    # ---------- Normal Equation ----------
    bias_ne, weights_ne = get_bias_and_weight(X, y, include_bias)
    pred_ne = X @ weights_ne + bias_ne
    mse_ne = mean_squared_error(y, pred_ne)

    # ---------- Find epochs needed for GD ----------
    num_of_epochs = 250
    if scaling:
        bias_tmp, weights_tmp, num_of_epochs, _ = find_number_of_epochs(
            X, y, lr, delta_loss
        )

    # ---------- Gradient Descent ----------
    bias_gd, weights_gd, loss = gradient_descent_multi_variable(
        X, y, lr=lr, number_of_epochs=num_of_epochs
    )
    pred_gd = X @ weights_gd + bias_gd
    mse_gd = mean_squared_error(y, pred_gd)


    # ---------- Comparison ----------
    return mse_ne, mse_gd
    print("\nNormal Equation:")
    print("MSE:", mse_ne)

    print("\nGradient Descent:")
    print("MSE:", mse_gd)
