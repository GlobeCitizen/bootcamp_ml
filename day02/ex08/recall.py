# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    recall.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/25 08:05:47 by mahnich           #+#    #+#              #
#    Updated: 2020/01/25 08:11:17 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def recall_score_(y_true, y_pred, pos_label=1):
    score = 0
    for x,y in zip(y_true, y_pred):
        if x == pos_label and y == pos_label:
            score += 1
    return score / np.count_nonzero(y_true == pos_label)


y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
print(recall_score_(y_true, y_pred))
y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog',
'dog', 'dog'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet',
'dog', 'norminet'])
print(recall_score_(y_true, y_pred, pos_label='dog'))
print(recall_score_(y_true, y_pred, pos_label='norminet'))
