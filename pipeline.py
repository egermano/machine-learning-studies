# Let’s Write a Pipeline - Machine Learning Recipes #4
# https://www.youtube.com/watch?v=84gqSbLcBFE&index=4&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

# print(predictions)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

print(accuracy_score(y_test, predictions))
