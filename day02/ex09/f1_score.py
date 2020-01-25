# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    f1_score.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/25 08:48:18 by mahnich           #+#    #+#              #
#    Updated: 2020/01/25 08:51:02 by mahnich          ###   ########.fr        #
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

def recall_score_(y_true, y_pred, pos_label=1):
    score = 0
    for x,y in zip(y_true, y_pred):
        if x == pos_label and y == pos_label:
            score += 1
    return score / np.count_nonzero(y_true == pos_label)

def f1_score_(y_true, y_pred, pos_label=1):
    recall = recall_score_(y_true, y_pred, pos_label)
    precision = precision_score_(y_true, y_pred, pos_label)
    return 2 * recall * precision /(recall + precision)


y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
print(f1_score_(y_true, y_pred))
# 0.5 # 0.5
# Test n.2
y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog',
'dog', 'dog'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet',
'dog', 'norminet'])
print(f1_score_(y_true, y_pred, pos_label='dog'))
# 0.6666666666666665
# 0.6666666666666665
# Test n.3
print(f1_score_(y_true, y_pred, pos_label='norminet'))
# 0.5714285714285715
# 0.5714285714285715
