import cv2
import os

for file_name in os.listdir('recordings'):
    file_path = os.path.join('recordings', file_name)
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        os.remove(file_path)
    else:
        cap.release()
