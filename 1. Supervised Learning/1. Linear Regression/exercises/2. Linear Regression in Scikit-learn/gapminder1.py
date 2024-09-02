# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
print(bmi_life_data.head())

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']].values, bmi_life_data['Life expectancy'])

# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict([[21.07931]])
print(laos_life_exp)


### Aditional test not part of Udacity
first_100 = bmi_life_data[:100]
X_train = first_100[['BMI']].values
Y_train = first_100['Life expectancy']

remaining = bmi_life_data[100:]
X_test = remaining[['BMI']].values
Y_test = remaining['Life expectancy']


model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

# Model coefficient
print('Coefficient: ', model.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred))

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_test, Y_pred, color="blue", linewidth=3)
plt.xticks(())
plt.yticks(())

plt.show()
