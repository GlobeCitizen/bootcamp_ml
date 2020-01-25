# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    vec_log_loss.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/22 19:55:42 by mahnich           #+#    #+#              #
#    Updated: 2020/01/25 07:15:07 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def sigmoid_(x):
    x0 = np.array(x)
    return 1 / (1 + np.exp(-x0))

def vec_log_loss_(y_true, y_pred, m, eps=1e-15):
    if m > 1:
        return -1 / m * np.dot(np.ones((1, m)), ((y_true * np.log(y_pred)) + (1 - y_true) * np.log(1- y_pred)))[0]
    else:
        return -1 / m * np.dot(np.ones((1, m)), ((y_true * np.log(y_pred)) + (1 - y_true) * np.log(1- y_pred)))[0][0]

x = np.array([1, 2, 3, 4])
y_true = 0
theta = np.array([-1.5, 2.3, 1.4, 0.7]) 
y_pred = sigmoid_(np.dot(x, theta)) 
m= 1
print(vec_log_loss_(y_true, y_pred, m)) # 10.100041078687479
