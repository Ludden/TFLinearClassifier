import tensorflow.contrib.learn as tf
from sklearn import datasets, metrics

iris = datasets.load_iris()

feature_columns = tf.infer_real_valued_columns_from_input(iris.data)
classifier = tf.LinearClassifier(feature_columns=feature_columns, n_classes=3)
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))

print "Accuracy: %f" % score
