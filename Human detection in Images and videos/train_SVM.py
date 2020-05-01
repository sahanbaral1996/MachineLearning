from sklearn import svm
from skimage.feature import hog
from sklearn.externals import joblib
import os
import numpy as np
import cv2

pos_im_path = 'photo/pos'
neg_im_path = 'photo/neg'

samples = []
test = []
labels = []

# Get positive samples
for filename in os.listdir(pos_im_path):
    print(filename)
    img = cv2.imread(os.path.join(pos_im_path,filename), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(64,128))
    hist =  hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2),transform_sqrt=False,feature_vector=True,visualize=False)
    samples.append(hist)
    labels.append(1)

# Get negative samples
for filename in os.listdir(neg_im_path):
    img = cv2.imread(os.path.join(neg_im_path,filename), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))
    hist = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L1',
                        transform_sqrt=False, feature_vector=True, visualize=False)
    samples.append(hist)
    labels.append(0)

# Convert objects to Numpy Objects

samples = np.float32(samples)
labels = np.array(labels)

# Shuffle Samples
rand = np.random.RandomState(321)
shuffle = rand.permutation(len(samples))
samples = samples[shuffle]
labels = labels[shuffle]

svm = svm.SVC(kernel='rbf', gamma=0.7, C=2)
svm.fit(samples,labels)

joblib.dump(svm,"model/svm_data.model")


