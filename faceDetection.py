import cv2
from imageUtil import *
import random
import os
import datetime
import time

def faceDetection(out_dir, image, image_size):
    haar = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    haar.load("./facePatten/haarcascade_frontalface_alt2.xml")
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray_img, 1.3, 5)
    if(len(faces) == 1):
        f_x, f_y, f_w, f_h = faces[0]
        face = image[f_y:f_y+f_h, f_x:f_x+f_w]
        face = cv2.resize(face, (image_size, image_size))
        #could deal with face to train
        face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
        now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(out_dir, str(now_time)+'.jpg')
        cv2.imwrite(image_path, face)
        return faces[0], image_path
    else:
        return "-","-"

if __name__ == '__main__':
    image = cv2.imread("./saveimage/blob.png")
    face, image_path = faceDetection("./faces/",image, 64)
    print(face)
    print(image_path)
    