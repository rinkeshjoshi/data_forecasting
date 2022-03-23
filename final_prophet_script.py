# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:05:36 2022

@author: rinke
"""

# Data Manipulation
import pandas as pd # for data manipulation
import numpy as np # for data manipulation

# Sklearn
from sklearn.linear_model import LinearRegression # for building a linear regression model
from sklearn.svm import SVR # for building SVR model
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

# Facebook Prophet
from prophet import Prophet

# Visualizations
import plotly.graph_objects as go # for data visualization
import plotly.express as px # for data visualization
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.io as pio
pio.renderers.default='browser'

file_name = str("data_forecasting/daily_ottawa_6105976_final_2021.csv")
data_frame = pd.read_csv(file_name)
data_frame = data_frame.set_index(["Year"])
#data_frame.head()

scaler = MinMaxScaler()
data_frame['Mean Temp (°C) (scaled)']=scaler.fit_transform(data_frame[['Mean Temp (°C)']])
data_frame['Date/Time'] = pd.to_datetime(data_frame['Date/Time'])
# print(data_frame.head())

# data frame for predictions
list_date = list(data_frame['Date/Time'])
list_temp = list(data_frame['Mean Temp (°C)'])
list_columns = {'ds': list_date, 'y': list_temp}
dataset = pd.DataFrame(data = list_columns)
print(dataset.head())

# dataset['ds','y'] = data_frame['Date/Time', 'Mean Temp (°C)']
model = Prophet()
model.fit(dataset)

# future periods to forecast for
future_predictions = model.make_future_dataframe(periods = 365)
print(future_predictions.tail())
print('-------------')
print(future_predictions.head())


# Predictions/Forecast
forecast_temp = model.predict(future_predictions)
print('-------------')
print(forecast_temp[['ds','yhat','yhat_lower','yhat_upper']].head())
print('-------------')
print(forecast_temp[['ds','yhat','yhat_lower','yhat_upper']].tail())

figure1 = model.plot(forecast_temp)
figure2 = model.plot_components(forecast_temp)

# plotly interactive graphs
plot_plotly(model, forecast_temp)
plot_components_plotly(model, forecast_temp)

