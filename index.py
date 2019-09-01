import os
from gevent import monkey
monkey.patch_all()
from flask import Flask, request
from gevent import pywsgi
import cv2
from faceDetection import faceDetection
import tensorflow as tf
from tensorflow import keras
from imageUtil import *
import numpy as np
import random
import json
import logging

IMGSIZE = 64
label_names = ['other', 'jessehe']

app = Flask(__name__)

@app.route('/')
def index():
    app.logger.info("my first logging")
    return 'Hello World'

@app.route('/faceDetection', methods=['POST'])
def faceDetectionApi():
    image_file = request.files['file']
    file_name = image_file.filename
    #file_path = './saveImage'
    app.logger.info(file_name)
    if image_file:
        # 地址拼接
        #file_paths = os.path.join(file_path, file_name)
        #app.logger.debug('file_paths:', file_name)
        # 保存接收的图片到文件夹下
        image_file.save(os.getcwd() + "/saveImage/"+ file_name)
        image = cv2.imread(os.getcwd() + "/saveImage/"+file_name)
        if(image is None):
            app.logger.info("/faceDetection | Error: image read error!")
            return "Error: image read error!"
    if image_file:
        face, faces_path = faceDetection(os.getcwd() + "/faces/",image, 64)
        if(face == "-"):
            return "error result!"
        result_dict = {}
        result_dict["image_path"] = faces_path
        result_dict["face_x"] = str(face[0])
        result_dict["face_y"] = str(face[1])
        result_dict["face_w"] = str(face[2])
        result_dict["face_h"] = str(face[3])
    return json.dumps(result_dict)

@app.route('/faceRecongnition', methods=['POST'])
def faceRecongnitionApi():
    image_file = request.files['file']
    file_name = image_file.filename
    result_list = []
    model = keras.models.load_model("/home/jessehe/Desktop/TF2.0/faceRecognition/model/keras-3.h5")
    if image_file:
        image_file.save(os.getcwd() + "/saveImage/"+ file_name)
        image = cv2.imread(os.getcwd() + "/saveImage/"+ file_name)
        if(image is None):
            app.logger.info("/faceRecongnition | Error: image read error!")
            return "Error: image read error!"
        _, face_path = faceDetection(os.getcwd() + "/faces/",image, 64)
        app.logger.info("faceRecongnition, face_path:" + face_path)
        if(face_path == "-"):
            return str("error result")
        face = load_and_preprocess_image(face_path)
        result = model.predict(tf.expand_dims(face,axis=0))
        result_list.append(label_names[np.argmax(result[0])])
    return str(result_list)

if __name__ == "__main__":
    app.debug = True
    app.run(port=9877)
