import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn import datasets
from mlxtend.data import loadlocal_mnist
train_images, train_labels = loadlocal_mnist(images_path='./train-images-idx3-ubyte/train-images-idx3-ubyte',
                                             labels_path='./train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images, test_labels = loadlocal_mnist(images_path='./t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
                                             labels_path='./t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
plt.imshow(train_images[25].reshape(28,28), cmap='gray')
train_labels[25]
feature, hog_img = hog(train_images[25].reshape(28,28), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2), visualize=True, block_norm='L2')
feature.shape
plt.bar(list(range(feature.shape[0])), feature)
n_dims = feature.shape[0]
n_samples = train_images.shape[0]
X_train, y_train = datasets.make_classification(n_samples=n_samples, n_features=n_dims)
X_train.shape
for i in range(n_samples):
    X_train[i], _ = hog(train_images[i].reshape(28,28), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2), visualize=True, block_norm='L2')
    y_train[i] = train_labels[i]
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(y_train)
y_train_one_hot = lb.transform(y_train)
y_train_one_hot[25]
y_train[25]

import numpy as np
label = lb.inverse_transform(np.array([y_train_one_hot[25]]))
label[0]
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
n_samples = test_images.shape[0]
X_test, y_test = datasets.make_classification(n_samples=n_samples, n_features=n_dims)
for i in range(n_samples):
    X_test[i], _ = hog(test_images[i].reshape(28,28), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2), visualize=True, block_norm='L2')
    y_test[i] = test_labels[i]
y_test_one_hot = lb.transform(y_test)
y_pred_one_hot = clf.predict(X_test)
import random
random_indices = random.sample(range(len(test_images)), 5)

plt.figure(figsize=(15, 3))
for i, index in enumerate(random_indices, 1):
    plt.subplot(1, 5, i)
    plt.imshow(test_images[index].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y_test[index]}")

plt.show()

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred_one_hot)
conf_mat
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=conf_mat, class_names=class_names)
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred_one_hot, average=None)
precision
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred_one_hot, average=None)
recall
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_one_hot)
accuracy
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred_one_hot, average='macro')
f1
