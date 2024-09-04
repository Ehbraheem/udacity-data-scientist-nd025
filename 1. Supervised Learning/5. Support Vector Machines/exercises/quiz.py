# Import statements
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data.
data = np.asarray(pd.read_csv("data.csv", header=None))
# Assign the features to the variable X, and the labels to the variable y.
X = data[:, 0:2]
y = data[:, 2]


def plot_data(X, y, title="Samples in two-dimensional feature space"):
    # Plotting settings
    fig, ax = plt.subplots(figsize=(4, 3))
    find_min = lambda g: np.min(g)
    find_max = lambda g: np.max(g)
    x_min, x_max, y_min, y_max = find_min(X), find_max(X), find_min(y), find_max(y)
    print("x_min = ", x_min, "x_max = ", x_max, "y_min = ", y_min, "y_max = ", y_max)
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot samples by color and add legend
    scatter = ax.scatter(X[:, 0], X[:, 1], s=150, c=y, label=y, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.set_title(title)
    _ = plt.show()


plot_data(X, y)

# TODO: Create the model and assign it to the variable model.
# Find the right parameters for this model to achieve 100% accuracy on the dataset.
model = SVC(kernel="rbf", gamma=26.5)

# TODO: Fit the model.
model.fit(X, y)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)
print("Accuracy score: ", acc * 100.0)


# Split data to properly test our algorithm
train_split = data[:70]
X_train = train_split[:, :2]
y_train = train_split[:, 2]

plot_data(X_train, y_train, title="Training data set")

test_split = data[70:]
X_test = test_split[:, :2]
y_test = test_split[:, 2]

plot_data(X_test, y_test, title="Test data set")

model = SVC(kernel="rbf", gamma=27)
clf = model.fit(X_train, y_train)

y_hat = model.predict(X_test)
accuracy = accuracy_score(y_test, y_hat)
print("Accuracy score: ", accuracy * 100)
