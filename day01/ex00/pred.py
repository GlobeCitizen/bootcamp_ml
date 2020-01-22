# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    pred.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/22 01:50:56 by mahnich           #+#    #+#              #
#    Updated: 2020/01/22 02:10:36 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
try:
    def predict_(theta, X):
        if X.shape[1] + 1 != theta.shape[0]:
            print('Incompatible dimension match between X and theta')
            return
        X = np.insert(X, 0, 1.0, axis=1)
        return np.dot(X, theta)
except:
    pass
