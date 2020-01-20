# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    mat_mat_prod.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/20 11:36:02 by mahnich           #+#    #+#              #
#    Updated: 2020/01/20 12:38:11 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def mat_mat_prod(x, y):
    if x.size == 0 or y.size == 0 or x.shape[1] != y.shape[0]:
        return
    z = np.zeros((x.shape[0], y.shape[1]), dtype=int)
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            for k in range(x.shape[1]):
                z[i][j] += x[i][k] * y[k][i]
    return z
