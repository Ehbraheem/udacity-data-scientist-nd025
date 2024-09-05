# Import, read, and split data
import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")

X = np.array(data[["x1", "x2"]])
y = np.array(data["y"])

# Fix random seed
np.random.seed(55)

### Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# TODO: Uncomment one of the three classifiers, and hit "Test Run"
# to see the learning curve. Use these to answer the quiz below.

### Logistic Regression
# estimator = LogisticRegression()

### Decision Tree
# estimator = GradientBoostingClassifier()

### Support Vector Machine
# estimator = SVC(kernel="rbf", gamma=1000)

from utils import draw_learning_curves

# draw_learning_curves(X, y, estimator, 10)

for estimator in [
    SVC(kernel="rbf", gamma=1000),
    GradientBoostingClassifier(),
    LogisticRegression(),
    DecisionTreeClassifier(),
]:
    draw_learning_curves(X, y, estimator, 10)
