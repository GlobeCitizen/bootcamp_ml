# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    log_gradient.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/22 17:08:30 by mahnich           #+#    #+#              #
#    Updated: 2020/01/25 02:58:34 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def sigmoid_(x):
    x0 = np.array(x)
    return 1 / (1 + np.exp(-x0))

def log_gradient_(x, y_true, y_pred):
    tmp = []
    try:
        for i in range(len(x)):
            tmp.append([(y_pred - y_true)[i] * x for x in x[i]])
        return [sum(j) for j in zip(*tmp)]
    except:
        return ([(y_pred - y_true) * x for x in x])

x = [1, -0.5, 2.3, -1.5, 3.2]
y_true = 0
theta = [0.5, -0.5, 1.2, -1.2, 2.3]
x_dot_theta = sum([a*b for a, b in zip(x, theta)])
y_pred = sigmoid_(x_dot_theta)
print(log_gradient_(x, y_true, y_pred))
