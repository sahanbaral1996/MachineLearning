import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os

X = []
Y = []
DIRS = ["sad-photos foldername","happy photos folder name"]
for index,dir in enumerate(DIRS):
    for images in os.listdir(dir+'/'):
        X.append(cv2.imread(os.path.join(dir,images),cv2.IMREAD_GRAYSCALE))
        Y.append(index)


# load data
X_train = np.array(X)
Y_train = np.array(Y)

print(X_train.shape)
print(Y_train.shape)

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2)

x = np.array(x_train).reshape(-1, 50, 50, 1)/255.0
x = np.float32(x)
y = np.array(y_train)

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, batch_size=64, epochs=10, validation_split=0.2)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('face.xml')
while True:
    res, Rimg = cap.read()
    prepare_img = cv2.cvtColor(Rimg,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(prepare_img, 1.3, 5)
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        cv2.rectangle(Rimg, (x, y), (x + w, y + h), (255, 0, 0), 2)
        img = Rimg[int(y):int(y) + int(h), int(x):int(x) + int(h)]
        send_img = cv2.resize(img,(50,50)).reshape(-1,50,50,1)
        if model.predict(send_img).any() > 0.5:
            cv2.putText(img, 'happy',(75,75),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1,cv2.LINE_AA)
        else:
            cv2.putText(img, 'sad', (75, 75), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('img',Rimg)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()



