import pandas as pd
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

pd.set_option('display.max_columns', 10)
dt = pd.read_csv("breast-cancer-wisconsin.data.txt")
dt.replace('?', -99999, inplace=True)
dt.drop(['id'], 1, inplace=True)

features = np.array(dt.drop(['class'], 1))
label = np.array(dt['class'])

features_train, features_test, label_train, label_test = train_test_split(features, label, test_size=0.2)
classifier = neighbors.KNeighborsClassifier()
classifier.fit(features_train, label_train)
accuracy = classifier.score(features_test, label_test)

print(accuracy)
#plt.scatter([4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1])
#plt.show()
example_measures = np.array([[8,10,1,10,8,10,3,2,1],[4,2,1,2,2,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = classifier.predict(example_measures)
print(prediction)
