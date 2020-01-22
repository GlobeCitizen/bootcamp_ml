# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    mylinearregression.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/22 03:27:53 by mahnich           #+#    #+#              #
#    Updated: 2020/01/22 14:29:22 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from numpy.linalg import inv

class MyLinearRegression:
    """
    Description:
        My personnal linear regression class to fit like a boss.
"""
    def __init__(self, theta):
        """
        Description:
            generator of the class, initialize self.
        Args:
            theta: has to be a list or a numpy array, it is a vector of
dimension (number of features + 1, 1).
        Raises:
            This method should noot raise any EXception.
        """
        self.theta = np.array(theta)

    def predict_(self, X):
        if X.shape[1] + 1 != self.theta.size:
            print('Incompatible dimension match between X and theta')
            return
        X = np.insert(X, 0, 1.0, axis=1)
        return np.dot(X, self.theta)

    def cost_elem_(self, X, Y):
        if X.shape[0] != Y.size:
            print('X and Y has incompatible dimensions')
            return
        if X.shape[1] + 1 != self.theta.size:
            print('Incmpatible dimension match betzeen X and theta')
            return
        m = X.shape[0]
        X = np.insert(X, 0, 1.0, axis=1)
        pred = np.dot(X, self.theta)
        return 0.5 * 1 / m * (pred - Y) ** 2

    def cost_(self, X, Y):
        return (float(sum(self.cost_elem_(X,Y))))

    def vec_gradient(self, X, Y):
        if X.size * Y.size * self.theta.size == 0 or X.shape[1] + 1 != self.theta.size or X.shape[0] != Y.size:
            return
        X = np.insert(X, 0, 1.0, axis = 1)
        return np.dot(np.transpose(X), np.dot(X, self.theta) - Y) / Y.size

    def fit_(self, X, Y, alpha, n_cycle):
        while n_cycle != 0:
            self.theta -= alpha * self.vec_gradient(X, Y)
            n_cycle -= 1
        return self.theta

    def mse_(self, x, y):
        if x.size * y.size * self.theta.size == 0 or x.shape[0] != y.size:
            return
        return sum((x - y) ** 2)[0] / y.size
    
    def normalequation_(self, X, Y):
        X = np.insert(X, 0, 1.0, axis=1)
        self.theta = inv(X.T.dot(X)).dot(X.T.dot(Y))
