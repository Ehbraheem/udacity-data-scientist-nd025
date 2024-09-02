# TODO: Add import statements
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('data.csv')
print(train_data.head())
X = train_data[:15][['Var_X']].values
y = train_data[:15]['Var_Y']

# Create polynomial features
# TODO: Create a PolynomialFeatures object, then fit and transform the
# predictor feature
poly_feat = PolynomialFeatures(degree=4)
X_poly = poly_feat.fit_transform(X)

# Make and fit the polynomial regression model
# TODO: Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = LinearRegression(fit_intercept = False).fit(X_poly, y)

# Once you've completed all of the steps, select Test Run to see your model
# predictions against the data, or select Submit Answer to check if the degree
# of the polynomial features is the same as ours!

# Test Run
X_test = train_data[15:][['Var_X']].values
y_test = train_data[15:]['Var_Y']
X_test_poly = poly_feat.fit_transform(X_test)

score = poly_model.score(X_test_poly, y_test)
print(score)
