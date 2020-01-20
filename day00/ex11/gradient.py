# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    gradient.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/20 12:58:00 by mahnich           #+#    #+#              #
#    Updated: 2020/01/20 14:37:39 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def gradient(x, y, theta):
    if x.size*y.size*theta.size == 0 or x.shape[1] != theta.size or x.shape[0] != y.size:
        return
    z = np.ndarray(theta.size)
    for j in range(len(theta)):
        res = 0
        for x_i, y_i in zip(x, y):
            res += np.dot(np.dot(theta, x_i) - y_i, x_i[j])
        z[j] = res
    return z / y.size
