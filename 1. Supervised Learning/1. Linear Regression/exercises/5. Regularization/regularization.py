# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import Lasso


# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('data.csv', header=None)
# print(train_data.head())
X = train_data.iloc[:, :6]
y = train_data.iloc[:, 6]

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()
lasso_reg.fit(X, y)

# TODO: Fit the model.


# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)
