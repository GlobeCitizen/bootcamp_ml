# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    mean.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/20 10:34:53 by mahnich           #+#    #+#              #
#    Updated: 2020/01/20 10:39:44 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def sum_(x, f):
    if x.size == 0:
        return
    res = 0
    for  y in x:
        res += f(y)

    return float(res)

def mean(x):
    if x.size == 0:
        return 
    return (sum_(x, lambda x: x) / x.size)
