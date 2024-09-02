# Import statements
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data.
data = np.asarray(pd.read_csv("data.csv", header=None))
# Assign the features to the variable X, and the labels to the variable y.
X = data[:, 0:2]
y = data[:, 2]

# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.
model = DecisionTreeClassifier()

# TODO: Fit the model.
fitted_tree = model.fit(X, y)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)
print("Accuracy score: ", acc * 100.0)


# Plot the decision tree
plt.figure()
plot_tree(fitted_tree, filled=True)
plt.title("Decision tree based on all the features of our data set")
plt.show()


y_pred_2D = np.reshape(y_pred, (y_pred.shape[0], -1))
print(y_pred.shape)
print(y_pred_2D.shape)
print(X[:, 0].shape)

# When you hit Test Run, you'll be able to see the boundary region of your model, which will help you tune the correct parameters, in case you need them.
display = DecisionBoundaryDisplay.from_estimator(
    fitted_tree,
    X,
    cmap=plt.cm.RdYlBu,
    response_method="predict",
    xlabel="X axis label",
    ylabel="Y axis label",
)
display.plot()

plt.show()
