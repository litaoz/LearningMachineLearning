# Recipe2.py
#
# Train on the Iris dataset: https://en.wikipedia.org/wiki/Iris_flower_data_set
# 
# Goals:
#  1. Import dataset
#  2. Train a classifier
#  3. Predict label for new flower
#  4. Visualize the tree
# 
# Tutorial on using machine learning from Google Developers
# https://www.youtube.com/watch?v=tNa99PG8hR8
#
# Additional comments and notes written by LZ
# Updated 10/1/17
#


# Step 1: Import dataset
from sklearn.datasets import load_iris
iris = load_iris()  # loads sklearn's iris dataset

# # get an idea of the dataset
# print(iris.feature_names)  # metadata of the features [len, width ... ]
# print(iris.target_names)
# print()
# print(iris.data[0])  # first entry in data (refer to feature_names for meaning)
# print(iris.target[0]) # first target, corresponds to data[0] (meaning in target_names)
# # ^0 = setosa

# Step 2: Train a classifier
import numpy as np  # import not in beginning to modularize learning 
from sklearn import tree

test_idx = [0, 50, 100]  # take out sample for testing later 
						 # Just so happens that the dataset is ordered
						 # so doing this takes one of each type of flower:
						 # 'setosa' 'versicolor' 'virginica'
# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# train data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)


# Step 3: Predict label for new flower
assert np.array_equal(
	clf.predict(test_data),
	test_target
	), "Expected [0, 1, 2] == [0, 1, 2]"


## Step 4: Vizualize the tree
## Note: video does not go into how it works
# from sklearn.externals.six import StringIO
# import pydot
# dot_data = StringIO()
# tree.export_graphviz(clf,
# 						out_file = dot_data,
# 						feature_names = iris.feature_names,
# 						class_names = iris.target_names,
# 						filled = True, rounded = True,
# 						impurity=False
# 					)

# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph[0].write_pdf("iris.pdf")  # Ran into problems here! 
# #TODO: evaluate "Excpetion: dot.exe not found in path"
