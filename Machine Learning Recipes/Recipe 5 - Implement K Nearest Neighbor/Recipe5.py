# Recipe5.py
#
# Write k-nearest neighbor classifier from scratch!
#
# Tutorial on using machine learning from Google Developers
# https://www.youtube.com/watch?v=AoeEHqVSNOw
#
# Additional comments and notes written by LZ
# Updated 10/1/17
#

from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a, b)

class ScrappyKNN():
    # Notice that we need at lease two methods at the minimum: fit and predict
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []    # x_test is a list of lists
        for row in x_test:  # each row is a data entry
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.x_train[0])
        best_index = 0
        for i, entry in enumerate(self.x_train):
            dist = euc(row, entry)
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]


############ Pipeline #####################
# Note: most of the code is from Recipe 4
###########################################
from sklearn import datasets
iris = datasets.load_iris()  # load the iris dataset, copvered in Recipe2

x = iris.data
y = iris.target

# Split up the training data from the test data 
# (Note: changed the import module because sk.cross_validation was being phased out)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

# Train K-nearest neighbor (Recipe 4)  
#from sklearn.neighbors import KNeighborsClassifier  # <- comment out to write our own!
#my_classifier = KNeighborsClassifier()
my_classifier = ScrappyKNN()

# Predict on test
my_classifier.fit(x_train, y_train)
predictions = my_classifier.predict(x_test)

# Obtain metrics
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))  # Note: I got tree:0.947, k-neighbor: 1.0
