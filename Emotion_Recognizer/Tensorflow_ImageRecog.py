import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation,Dropout,BatchNormalization
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras import utils
import cv2
import os


df = pd.read_csv("C:\\Users\\sahan.HP2_NOV2018\\Documents\\fer\\fer2013\\fer2013.csv")

#train_df = df[['Usage']=='Training']


X_train,train_y,X_test,test_y=[],[],[],[]

for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
           X_train.append(np.array(val,'float32'))
           train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
           X_test.append(np.array(val,'float32'))
           test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")

X_train = np.array(X_train,'float32')/255
train_y = np.array(train_y,'float32')
X_test = np.array(X_test,'float32')/255
test_y = np.array(test_y,'float32')

train_y=utils.to_categorical(train_y, num_classes=7)
test_y=utils.to_categorical(test_y, num_classes=7)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

model = Sequential()

model.add(Conv2D(64, kernel_size=(3,3),activation='relu',kernel_initializer='uniform',input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, kernel_size=(3,3),activation='relu',kernel_initializer='uniform',input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, kernel_size=(3,3),activation='relu',kernel_initializer='uniform',input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=(3,3),activation='relu',kernel_initializer='uniform',input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2) )
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Training the model
model.fit(X_train, train_y,
          batch_size=64,
          epochs=15,
          verbose=1,
          validation_data=(X_test, test_y),
          shuffle=True)

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



