# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    vec_gradient.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/20 13:27:50 by mahnich           #+#    #+#              #
#    Updated: 2020/01/20 13:31:24 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def vec_gradient(x, y, theta):
    if x.size*y.size*theta.size == 0 or x.shape[1] != theta.size or x.shape[0] != y.size:
        return
    return np.dot(np.transpose(x),np.dot(x, theta) - y) / y.size
