# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    linear_mse.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/20 12:39:24 by mahnich           #+#    #+#              #
#    Updated: 2020/01/20 12:48:15 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def linear_mse(x, y, theta):
    if x.size*y.size*theta.size == 0 or x.shape[1] != theta.size or x.shape[0] != y.size:
        return
    res = 0
    for x_i, y_i in zip(x, y):
        res += (np.dot(theta, x_i) - y_i) ** 2
    return res / y.size
