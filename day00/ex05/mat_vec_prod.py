# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    mat_vec_prod.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/20 11:02:21 by mahnich           #+#    #+#              #
#    Updated: 2020/01/20 12:01:23 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def dot(x, y):
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return
    res = 0
    for x_i, y_i in zip(x, y):
        res += x_i * y_i
    return (float(res))

def mat_vec_prod(x, y):
    if x.size == 0 or y.size == 0 or x.shape[1] != y.size:
        return
    z = np.ndarray((x.shape[0],1), dtype=int)
    for i, x_i in zip(range(x.shape[0]), x):
        z[i] = dot(x_i, y)
    return z
