# Recipe4.py
#
# Pipelines! Note that there are two different classifiers that can
# be switched around in the code below.
#
# Tutorial on using machine learning from Google Developers
# https://www.youtube.com/watch?v=84gqSbLcBFE
#
# Additional comments and notes written by LZ
# Updated 10/1/17
#

from sklearn import datasets
iris = datasets.load_iris()  # load the iris dataset, copvered in Recipe2

x = iris.data
y = iris.target

# Split up the training data from the test data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)


# Option 1: Train tree
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

# # Option 2: Train K-nearest neighbor
# from sklearn.neighbors import KNeighborsClassifier
# my_classifier = KNeighborsClassifier()

# Predict on test
my_classifier.fit(x_train, y_train)
predictions = my_classifier.predict(x_test)

# Obtain metrics
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))  # Note: I got tree:0.947, k-neighbor: 1.0
