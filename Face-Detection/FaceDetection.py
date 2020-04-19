import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('face.xml')

photos_dir = 'dirctory to yout photos'

faced_photo_dir = 'directory to add photo face'

for img_name in os.listdir("photos_dir/"):
    gray = os.path.join("photos_dir/"+img_name)
    img = cv2.imread(gray)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
        img = img[int(y):int(y)+int(h), int(x):int(x)+int(h)]
        try:
            cv2.imwrite("faced_photo_dir/"+img_name,cv2.resize(img,(50,50))) # resized to 50,50
        except Exception:
            pass