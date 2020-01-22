# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    mylinearregression.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/22 03:27:53 by mahnich           #+#    #+#              #
#    Updated: 2020/01/22 09:05:24 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

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
        if X.shape[1] + 1 != self.theta.shape[0]:
            print('Incompatible dimension match between X and theta')
            return
        X = np.insert(X, 0, 1.0, axis=1)
        return np.dot(X, self.theta)

    def cost_elem_(self, X, Y):
        if X.shape[0] != Y.shape[0]:
            print('X and Y has incompatible dimensions')
            return
        if X.shape[1] + 1 != self.theta.shape[0]:
            print('Incmpatible dimension match betzeen X and theta')
            return
        m = X.shape[0]
        X = np.insert(X, 0, 1.0, axis=1)
        pred = np.dot(X, self.theta)
        return 0.5 / m * (pred - Y) ** 2

    def cost_(self, X, Y):
        return (float(sum(self.cost_elem_(X,Y))))

    def vec_gradient(self, X, Y):
        if X.size * Y.size * self.theta.size == 0 or X.shape[1] + 1 != self.theta.shape[0] or X.shape[0] != Y.size:
            return
        X = np.insert(X, 0, 1.0, axis = 1)
        return np.dot(np.transpose(X), np.dot(X, self.theta) - Y) / Y.size

    def fit_(self, X, Y, alpha, n_cycle):
        while n_cycle != 0:
            self.theta -= alpha * self.vec_gradient(X, Y)
            n_cycle -= 1
        return self.theta

    def mse_(self, x, y):
        if x.size * y.size * self.theta.size == 0 or x.shape[1] + 1 != self.theta.shape[0] or x.shape[0] != y.size:
            return

        return sum((x - y) ** 2)[0] / y.size

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89.,
144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])
print(mylr.predict_(X))
print(mylr.cost_elem_(X,Y))
print(mylr.cost_(X,Y))
mylr.fit_(X, Y, alpha = 1.6e-4, n_cycle=200000)
print(mylr.theta)
print(mylr.predict_(X))
print(mylr.cost_elem_(X,Y))
print(mylr.cost_(X,Y))
