# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    confusion_matrix.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/25 08:17:06 by mahnich           #+#    #+#              #
#    Updated: 2020/01/25 10:24:09 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def confusion_matrix_(y_true, y_pred, labels=None):
    unique = sorted(set(y_true))
    unique2 = sorted(set(y_pred))
    dim = max(len(unique), len(unique2))
    matrix = np.zeros((dim, dim))
    if len(unique) > len(unique2):
        imap = {key: i for i, key in enumerate(unique)}
    else:
        imap = {key: i for i, key in enumerate(unique2)}
    # Generate Confusion Matrix
    for a,p in zip(y_true, y_pred):
        matrix[imap[a]][imap[p]] += 1

    if labels==None:
        return matrix
    else:
        i = imap[labels[0]]
        j = imap[labels[1]]

        return matrix[i:,i:j]
y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog',
'bird'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet'])
print(confusion_matrix_(y_true, y_pred))
print(confusion_matrix_(y_true, y_pred, labels=['bird','dog']))
