# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    segmoid.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/22 15:08:14 by mahnich           #+#    #+#              #
#    Updated: 2020/01/22 15:55:20 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def sigmoid_(x):
    x0 = np.array(x)
    return 1 / (1 + np.exp(-x0))

x = -4
print(sigmoid_(x))
# 0.01798620996209156
x= 2
print(sigmoid_(x))
# 0.8807970779778823
x = [-4, 2, 0]
print(sigmoid_(x))
# [0.01798620996209156, 0.8807970779778823, 0.5]
