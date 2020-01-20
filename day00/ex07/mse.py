# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    mse.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/20 11:42:12 by mahnich           #+#    #+#              #
#    Updated: 2020/01/20 12:27:36 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def mse(y, y_hat):
    if y.size == 0 or y_hat.size == 0 or y.size != y_hat.size:
        return
    res = 0
    for y_i, y_hat_i in zip(y, y_hat):
        res += (y_hat_i - y_i) ** 2
    return res / y.size
