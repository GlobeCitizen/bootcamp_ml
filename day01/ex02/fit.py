# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    fit.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/22 02:38:03 by mahnich           #+#    #+#              #
#    Updated: 2020/01/22 09:39:53 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

try:
    def vec_gradient(x, y, theta):
        if x.size*y.size*theta.size == 0 or x.shape[1] + 1 != theta.shape[0] or x.shape[0] != y.size:
            return
        x = np.insert(x, 0, 1.0, axis = 1)
        return np.dot(np.transpose(x),np.dot(x, theta) - y) / y.size

    def fit_(theta, X, Y, alpha, n_cycle):
        while n_cycle != 0:
            theta -= alpha * vec_gradient(X, Y, theta)
            n_cycle -= 1
        return theta
except:
    pass
