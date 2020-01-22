# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    multi_linear_model.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/22 08:25:34 by mahnich           #+#    #+#              #
#    Updated: 2020/01/22 13:54:18 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR

data = pd.read_csv('../resources/spacecraft_data.csv')

X1 = np.array(data['Age']).reshape(-1,1)
X2 = np.array(data['Thrust_power']).reshape(-1,1)
X3 = np.array(data['Terameters']).reshape(-1,1)

Y = np.array(data['Sell_price']).reshape(-1,1)

linear_model = MyLR([[1000.0],[-1.0]])
linear_model2 =  MyLR([[0.0],[1.0]])


linear_model.fit_(X1, Y, alpha=0.0005, n_cycle=100000)
Y_model_age = linear_model.predict_(X1)
plt.subplot(131)
data_plot = plt.plot(X1, Y, 'bo', label='Sell price')
predict_plot = plt.plot(X1, Y_model_age, 'c.', label='Predicted sell price')
plt.xlabel('age (years)')
plt.ylabel('sell price (keuros)')
plt.legend()
plt.grid()

linear_model2.fit_(X2, Y, alpha=0.00005, n_cycle=100000)
Y_model_thrust = linear_model2.predict_(X2)
plt.subplot(132)
data_plot = plt.plot(X2, Y, 'go', label='Sell price')
predict_plot = plt.plot(X2, Y_model_thrust, '.', color='lime', label='Predicted sell price')
plt.xlabel('Thrust power (10Km/s)')
plt.ylabel('sell price (keuros)')
plt.legend()
plt.grid()

linear_model.fit_(X3, Y, alpha=0.00005, n_cycle=100000)
Y_model_distance = linear_model.predict_(X3)
plt.subplot(133)
data_plot = plt.plot(X3, Y, 'mo', label='Sell price')
predict_plot = plt.plot(X3, Y_model_distance, '.', color='violet', label='Predicted sell price')
plt.xlabel('distance (Terameters))')
plt.ylabel('sell price (keuros)')
plt.legend()
plt.grid()

plt.show()

print(linear_model.mse_(Y_model_age, Y))

X = np.array(data[['Age','Thrust_power','Terameters']])

my_lreg = MyLR([[1000.0], [0.0], [1000.0], [1.0]])

my_lreg.fit_(X, Y, alpha=0.00005, n_cycle=125)
Y_multi_model = my_lreg.predict_(X)

plt.subplot(131)
data_plot = plt.plot(X1, Y, 'o', color='darkblue', label='Sell price')
plt.plot(X1, Y_multi_model, 'c.')
plt.grid()

plt.subplot(132)
data_plot = plt.plot(X2, Y, 'go', label='Sell price')
plt.plot(X2, Y_multi_model, '.', color='lime')
plt.grid()

plt.subplot(133)
data_plot = plt.plot(X3, Y, 'mo', label='Sell price')
plt.plot(X3, Y_multi_model, '.', color='violet')
plt.grid()

plt.show()

print(my_lreg.cost_(Y_multi_model, Y))
