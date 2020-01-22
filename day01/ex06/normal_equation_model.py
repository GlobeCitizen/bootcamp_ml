# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    normal_equation_model.py                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/22 11:54:18 by mahnich           #+#    #+#              #
#    Updated: 2020/01/22 14:55:51 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR

data = pd.read_csv("../resources/spacecraft_data.csv")

X = np.array(data[['Age','Thrust_power','Terameters']])
Y = np.array(data['Sell_price']).reshape(-1,1)

myLR_ne = MyLR([[1.0], [1.0], [1.0], [1.0]])
myLR_lgd = MyLR([[1.0], [1.0], [1.0], [1.0]])

myLR_lgd.fit_(X,Y, alpha = 5e-5, n_cycle = 100000)
Y_lgr_model = myLR_lgd.predict_(X)
myLR_ne.normalequation_(X,Y)
Y_ne_model = myLR_ne.predict_(X)
print(myLR_lgd.mse_(Y_lgr_model,Y))
print(myLR_ne.mse_(Y_ne_model,Y))

X1 = np.array(data['Age']).reshape(-1,1)

data_plot = plt.plot(X[:,0], Y, 'bo', label='Predicted sell price')
ne_plot = plt.plot(X1, Y_lgr_model, '.', color='orange', label='Prediction with LGD')
lgd_plot = plt.plot(X1, Y_ne_model, '.', color='springgreen',label='Prediction with NE')
plt.xlabel('age (Years)')
plt.ylabel('Selling Price (Keuros)')
plt.legend(loc=0)
plt.grid()
plt.show()
