# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    vec_linear_mse.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/20 12:49:06 by mahnich           #+#    #+#              #
#    Updated: 2020/01/20 12:53:36 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def vec_linear_mse(x, y , theta):
    if x.size*y.size*theta.size == 0 or x.shape[1] != theta.size or x.shape[0] != y.size:
        return
    return np.dot(np.dot(x, theta) - y, np.transpose(np.dot(x, theta) - y)) / y.size
