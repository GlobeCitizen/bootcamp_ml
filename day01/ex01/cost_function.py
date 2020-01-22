# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    cost_function.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/22 02:11:26 by mahnich           #+#    #+#              #
#    Updated: 2020/01/22 02:37:34 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

try:
    def cost_elem_(theta, X, Y):
        if X.shape[0] != Y.shape[0]:
            print('X and Y has incompatible dimensions')
            return
        if X.shape[1] + 1 != theta.shape[0]:
            print('Incmpatible dimension match betzeen X and theta')
            return
        m = X.shape[0]
        X = np.insert(X, 0, 1.0, axis=1)
        pred = np.dot(X, theta)
        return 0.5 * 1 / m * (pred - Y) ** 2

    def cost_(theta, X, Y):
        return (float(sum(cost_elem_(theta, X, Y))))
except:
    pass
