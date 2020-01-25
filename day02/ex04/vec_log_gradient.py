# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    vex_log_gradient.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/25 03:02:43 by mahnich           #+#    #+#              #
#    Updated: 2020/01/25 03:07:59 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def sigmoid_(x):
    x0 = np.array(x)
    return 1 / (1 + np.exp(-x0))

def vec_log_gradient_(x, y_true, y_pred):
    return np.dot((y_pred - y_true), x)

x = np.array([1, -0.5, 2.3, -1.5, 3.2]) # x[0] represent the intercept
y_true = 0
theta = np.array([0.5, -0.5, 1.2, -1.2, 2.3])
y_pred = sigmoid_(np.dot(x, theta))
print(vec_log_gradient_(x, y_true, y_pred))
# [ 0.99999686 -0.49999843 2.29999277 -1.49999528 3.19998994]

