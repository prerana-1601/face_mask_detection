# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 11:07:37 2020

@author: radha
"""
#This code ->runs through the folder which contains the images identifies the face(using MTCNN) 
#->the output will contain only the resized lower half of the face  

import cv2
import os
import glob
import cv2
import mtcnn
from mtcnn import MTCNN

detector = MTCNN()
count=0
#img_dir should contain the file location which contains all the images of faces with and without mask
img_dir =r"C:\Users\radha\Documents\face_mask\large\without_mask" 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    count=count+1
    faces = detector.detect_faces(img)
    for f in faces:
        x, y, w, h = f['box']
        #Save just the lower face
        face_img = img[(y+int(h/2)):y+h, x:x+w]
        face_img=cv2.resize(face_img,50,50)
        #to check if the face has been segmented correctly
        #cv2.imshow('current',face_img)
        cv2.imwrite(r"C:\Users\radha\Documents\face_mask\large\no_mask\frame%d.jpg" % count,face_img)
        