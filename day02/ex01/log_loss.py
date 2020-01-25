# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    log_loss.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/22 15:57:59 by mahnich           #+#    #+#              #
#    Updated: 2020/01/22 17:06:28 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #
import numpy as np
from math import log
def sigmoid_(x):
    x0 = np.array(x)
    return 1 / (1 + np.exp(-x0))

def log_loss_(y_true, y_pred, m, eps=1e-15):
    res = 0
    if m > 1:
        for y,y_ in zip(y_true, y_pred):
            res += y * log(y_) + (1 - y) * log(1 - y_)
        return - 1 / m * res
    else:
            return -1 / m *(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))



x_new = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
y_true = [1, 0, 1]
theta = [-1.5, 2.3, 1.4, 0.7]
x_dot_theta = x_new.dot(theta)
y_pred = sigmoid_(x_dot_theta)
m = len(y_true)
print(log_loss_(y_true, y_pred, m))
