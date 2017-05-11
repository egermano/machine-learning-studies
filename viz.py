# Visualizing a Decision Tree - Machine Learning Recipes #2
# https://www.youtube.com/watch?v=tNa99PG8hR8&index=2&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal


import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

# Print Data
# print (iris.feature_names)
# print (iris.target_names)
# print (iris.data[0])
# print (iris.target[0])
#
#
# for i in range(len(iris.target)):
#     print ("Example %d: lavel %s, features %s" % (i, iris.target[i], iris.data[i]))

# Simple Teste
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))


# viz code

from sklearn.externals.six import StringIO
import pydot

dot_data = StringIO()

tree.export_graphviz(
    clf,
    out_file=dot_data,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True, rounded=True,
    impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("iris.pdf")