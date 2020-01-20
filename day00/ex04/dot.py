# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    dot.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/20 10:53:18 by mahnich           #+#    #+#              #
#    Updated: 2020/01/20 11:53:54 by mahnich          ###   ########.fr        #
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
