# import libraries for Holt's linear model
import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt
from statsmodels.tsa.api import Holt
from sklearn.metrics import mean_squared_error 
from math import sqrt 

# import dataset, split into train and test
series = read_csv('airlines.csv')
train = series.iloc[0:22]
test = series.iloc[22:]

# Fitting the model to Holt's linear model and forecast the test set
y_hat = test.copy()
fit = Holt(np.asanyarray(train['passengers'])).fit(optimized=True)
y_hat['Holt_linear'] = fit.forecast(len(test))
y_hat

# plot the forecast
plt.figure(figsize=(10,3))
plt.plot(train['passengers'], label='train')
plt.plot(test['passengers'], label='test')
plt.plot(y_hat['Holt_linear'], label='Holt linear')
plt.legend(loc='best')
plt.show()

# compute the error
rmse=sqrt(mean_squared_error(test.passengers, y_hat['Holt_linear']))
print(rmse)


#---------------------------------------------------------------


# damped trend (optional)
fit = Holt(np.asanyarray(train['passengers']), damped=True).fit(optimized=True)
y_hat['Holt_linear_damped'] = fit.forecast(len(test))
y_hat
plt.figure(figsize=(10,3))
plt.plot(train['passengers'], label='train')
plt.plot(test['passengers'], label='test')
plt.plot(y_hat['Holt_linear'], label='Holt linear')
plt.plot(y_hat['Holt_linear_damped'], label='Holt linear damped')
plt.legend(loc='best')
plt.show()

rmse=sqrt(mean_squared_error(test.passengers, y_hat['Holt_linear_damped']))
print(rmse)

