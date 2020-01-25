# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    precision.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/25 08:42:31 by mahnich           #+#    #+#              #
#    Updated: 2020/01/25 08:47:20 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def precision_score_(y_true, y_pred, pos_label=1):
    TP = 0
    FP = 0
    for y_t, y_p in zip(y_true, y_pred):
        if y_t == pos_label and y_p == pos_label:
            TP += 1
        if y_t != pos_label and y_p == pos_label:
            FP += 1
    return TP / (TP + FP)

y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
print(precision_score_(y_true, y_pred))
# 0.4 # 0.4
# Test n.2
y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog',
'dog', 'dog'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet',
'dog', 'norminet'])
print(precision_score_(y_true, y_pred, pos_label='dog'))
# 0.6
# 0.6
# Test n.3
print(precision_score_(y_true, y_pred, pos_label='norminet'))
