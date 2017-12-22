# Recipe1.py
# 
# Categorizes apple and oranges based on features using a decision tree
# Info provided in the table below:
#
# | Weight | Texture | Label  |
# |--------|---------|--------|
# | 150g   | Bumpy   | Orange |
# | 170g   | Bumpy   | Orange |
# | 140g   | Smooth  | Apple  |
# | 130g   | Smooth  | Apple  |
#
# Tutorial on using machine learning from Google Developers
# https://www.youtube.com/watch?v=cKxRvEZd3Mw
#
# Additional comments and notes written by LZ
# Updated 10/1/17
#


from sklearn import tree

# Supervised Learning Recipe 
# Step 1: Collect Training Data
features = [
    [150, 0],  # [weight(g), smooth(bool)]
    [170, 0],
    [140, 1],
    [130, 1]
    ]
labels = [0, 0, 1, 1]  # Labels, 0 = orange, 1 = apple

# Step 2: Train Classifer
# For this tutorial we will use "Decision Tree"
clf = tree.DecisionTreeClassifier()  # variable name is "classifer"
clf = clf.fit(features, labels)

# Step 3: Make Predictions
# Predict an orange-like (heavier and bumpy)
print(clf.predict([[160, 0]]))
assert clf.predict([[160, 0]]) == [0], "Should predict an orange"
