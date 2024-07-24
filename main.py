import os
import pickle

import numpy as np

from skimage.io import imread
from skimage.transform import resize 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

input_dir = "data/clf-data"
categories = os.listdir(input_dir)

data = []
labels = []

# Prepare data
for category_idx, category in enumerate(categories):
    category_dir = os.path.join(input_dir, category)
    images = os.listdir(category_dir)
    for image in images:
        image_path = os.path.join(category_dir, image)
        img = imread(image_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)

print("Categories: ", categories)

data = np.asarray(data)
labels = np.asarray(labels)

# train / test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, shuffle=True)

# train classifier
classifiers = SVC()
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifiers, parameters)
grid_search.fit(X_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(X_test)
score = accuracy_score(y_prediction, y_test)

print("{}% of samples were correctly classified".format(score * 100))

pickle.dump(best_estimator, open("./model.p", "wb"))
