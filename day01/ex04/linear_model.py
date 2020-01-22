# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    linear_model.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/22 04:08:47 by mahnich           #+#    #+#              #
#    Updated: 2020/01/22 08:13:28 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR

data = pd.read_csv("../resources/are_blue_pills_magics.csv")
linear_model1 = MyLR([[89.0], [-8]])
linear_model2 = MyLR([[89.0], [-6]])

Xpill = np.array(data['Micrograms']).reshape(-1,1)
Yscore = np.array(data['Score']).reshape(-1,1)

linear_model1 = MyLR(np.array([[89.0], [-8]]))
linear_model2 = MyLR(np.array([[89.0], [-6]]))
Y_model1 = linear_model1.predict_(Xpill)
Y_model2 = linear_model2.predict_(Xpill)

print(linear_model1.mse_(Y_model1, Yscore))
linear_model1.fit_(Xpill, Yscore, alpha=0.0005, n_cycle=10000)

Y_model1 = linear_model1.predict_(Xpill)
print(linear_model1.mse_(Y_model1, Yscore))
plot1 = plt.plot(Xpill, Yscore, 'co', label='$S_{score}(pills)$')
plot2 = plt.plot(Xpill, Y_model1, 'g--o', label='$S_{predict}(pills)$')
plt.ylabel('Space driving score')
plt.xlabel('Quantity of blue pills (Mg)')
plt.legend(loc=0)
plt.grid()
plt.show()

theta0 = [89, 90, 85, 95, 87, 86]
theta1 = np.arange(-14,-4, 0.01)
linear_models = np.zeros((len(theta0),len(theta1)))
for c in range(len(theta0)):
    for i,theta in enumerate(theta1):
        linear_models[c][i] = MyLR([[theta0[c]],[theta]]).cost_(Xpill, Yscore)
for theta0,model in zip(theta0,linear_models):
    plt.plot(theta1, model, label=r'$J((\theta_0={}, \theta_1)$'.format(theta0))
plt.legend(loc=0)
plt.autoscale(axis='both')
plt.show()
