# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    log_reg.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mahnich <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/25 03:13:14 by mahnich           #+#    #+#              #
#    Updated: 2020/01/25 07:58:54 by mahnich          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #
import numpy as np
import pandas as pd
from math import log
import matplotlib.pyplot as plt


class LogisticRegressionBatchGd:
    def __init__(self, alpha=0.001, max_iter=1000, verbose=False, learning_rate='constant'):
        self.alpha = alpha
        self.max_iter = max_iter
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.thetas = []
        self.loss = []

    def __sigmoid_(self, x):
        x0 = np.array(x)
        return 1 / (1 + np.exp(-x0))
    
    def __vec_log_loss_(self, y_true, y_pred, m, eps=1e-15):
        if m > 1:
            return -1 / m * np.dot(np.ones((1, m)), ((y_true * np.log(y_pred)) + (1 - y_true) * np.log(1- y_pred)))[0]
        else:
            return -1 / m * np.dot(np.ones((1, m)), ((y_true * np.log(y_pred)) + (1 - y_true) * np.log(1- y_pred)))[0][0]
    def __vec_log_gradient_(self, x, y_true, y_pred):
            return np.dot(x.T, (y_pred - y_true)) / y_train.size

    def fit(self, x_train, y_train):
        for i in range(self.max_iter + 1):
            z = np.dot(x_train, self.thetas)
            h = self.__sigmoid_(z)
            gradient = np.dot(x_train.T, (h - y_train)) / y_train.size
            self.thetas -= self.alpha * self.__vec_log_gradient_(x_train, y_train, h)
            loss = self.__vec_log_loss_(y_train, h, len(y_train))
            self.loss.append(loss)
 
            if(self.verbose == True and i % 150 == 0):
                z = np.dot(x_train, self.thetas)
                h = self.__sigmoid_(z)
                print('epoch {} : loss {}'.format(i, loss))

    def predict(self, x):
        z = np.dot(x, self.thetas)
        return self.__sigmoid_(z) >= 0.5

    def score(self, x, y):
        y_pred = self.predict(x)
        score = 0
        return (y_pred == y).mean()


def accuracy_score_(y_true, y_pred):
    return (y_true == y_pred).mean()


from sklearn.metrics import accuracy_score

df_train = pd.read_csv('../resources//train_dataset_clean.csv', delimiter=',',
header=None, index_col=False)
x_train, y_train = np.array(df_train.iloc[:, 1:82]), df_train.iloc[:, 0]
df_test = pd.read_csv('../resources/test_dataset_clean.csv', delimiter=',', header=None,
index_col=False)
x_test, y_test = np.array(df_test.iloc[:, 1:82]), df_test.iloc[:, 0]
x_train = np.insert(x_train, 0, 1.0, axis=1)

x_test = np.insert(x_test, 0, 1.0, axis=1)
# We set our model with our hyperparameters : alpha, max_iter, verbose and learning_rate
model = LogisticRegressionBatchGd(alpha=0.1, max_iter=1500, verbose=True, learning_rate='constant')
# We fit our model to our dataset and display the score for the train and test datasets
model.thetas = np.ones(x_train.shape[1])
model.fit(x_train, y_train)
print(f'Score on train dataset : {model.score(x_train, y_train)}')
y_pred = model.predict(x_test)
print(f'Score on test dataset : {(y_pred == y_test).mean()}')
print(accuracy_score_(y_test, y_pred))
plt.plot(range(model.max_iter + 1), model.loss)
plt.show()

