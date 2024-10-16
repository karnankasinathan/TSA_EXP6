### Name:Karnan K
### Reg no:212222230062
### Date:
# Ex.No: 6               HOLT WINTERS METHOD




### AIM:
To implement the Holt Winters Method Model using Python.
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import math

# Load the dataset
df = pd.read_csv('/content/baggagecomplaints.csv', parse_dates=['Date'], index_col='Date')

# Preview the data
print(df.head())

# Group and resample the data to monthly frequency
df_monthly = df.resample('M').sum()

# Plot the time series
df_monthly['Baggage'].plot(figsize=(10,6), title="Monthly Airline Passenger Baggage Complaints")
plt.show()

decomposition = seasonal_decompose(df_monthly['Baggage'], model='additive')
fig = decomposition.plot()
plt.show()

# Calculate the mean and standard deviation
mean_complaints = df_monthly['Baggage'].mean()
std_complaints = df_monthly['Baggage'].std()

print(f"Mean: {mean_complaints}, Standard Deviation: {std_complaints}")

train_data = df_monthly.iloc[:-12]  # Use all data except the last 12 months for training
test_data = df_monthly.iloc[-12:]   # The last 12 months as the test set

# Fit the Holt-Winters Model
hw_model = ExponentialSmoothing(train_data['Baggage'], 
                                seasonal='add', trend='add', seasonal_periods=12).fit()

# Forecast future values for the test set period
forecast = hw_model.forecast(steps=12)

# Plot the original data and the predictions
plt.figure(figsize=(10,6))
plt.plot(train_data.index, train_data['Baggage'], label='Train Data')
plt.plot(test_data.index, test_data['Baggage'], label='Test Data')
plt.plot(forecast.index, forecast, label='Holt-Winters Forecast', color='red')
plt.title("Holt-Winters Forecast vs Real Data")
plt.legend()
plt.show()
# Calculate the RMSE for the test period
rmse = math.sqrt(mean_squared_error(test_data['Baggage'], forecast))
print(f"RMSE: {rmse}")

# Forecast for the next 12 months
future_forecast = hw_model.forecast(steps=12)

# Plot future forecast along with past data
plt.figure(figsize=(10,6))
plt.plot(df_monthly.index, df_monthly['Baggage'], label='Original Data')
plt.plot(future_forecast.index, future_forecast, label='Future Forecast', color='green')
plt.title("Future Forecast of Airline Passenger Baggage Complaints")
plt.legend()
plt.show()
```

### OUTPUT:


TEST_PREDICTION

![Screenshot 2024-10-16 093335](https://github.com/user-attachments/assets/607ed50f-66cc-42c5-988f-1bf25d310102)


FINAL_PREDICTION

![Screenshot 2024-10-16 093349](https://github.com/user-attachments/assets/4f210d75-d804-46d1-9bdb-dacd7931d167)

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
