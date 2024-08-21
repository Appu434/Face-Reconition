import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier('D:/MY_PROGRAMS/PYTHON/Face Recognition/data/haarcascade_frontalface_default.xml')

faces_data = []

i = 0

name = input("Enter Your Name: ")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(len(faces_data), -1)

if 'names.pkl' not in os.listdir('D:/MY_PROGRAMS/PYTHON/Face Recognition/data/'):
    names = [name] * len(faces_data)
    with open('D:/MY_PROGRAMS/PYTHON/Face Recognition/data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('D:/MY_PROGRAMS/PYTHON/Face Recognition/data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * len(faces_data)
    with open('D:/MY_PROGRAMS/PYTHON/Face Recognition/data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir('D:/MY_PROGRAMS/PYTHON/Face Recognition/data/'):
    with open('D:/MY_PROGRAMS/PYTHON/Face Recognition/data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('D:/MY_PROGRAMS/PYTHON/Face Recognition/data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    # Ensure the new data has the same shape as the existing data
    if faces.shape[1] == faces_data.shape[1]:
        faces = np.append(faces, faces_data, axis=0)
    else:
        print(f"Error: The shapes of the arrays do not match. Existing data shape: {faces.shape}, new data shape: {faces_data.shape}")
    with open('D:/MY_PROGRAMS/PYTHON/Face Recognition/data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)

print(f"Total face samples: {len(faces)}")
print(f"Total labels: {len(names)}")
