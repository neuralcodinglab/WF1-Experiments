import numpy as np
import cv2
import torch
import improc
import time
import pyduino
# import win32api
import csv
from csv import reader
import pandas as pd
import os
# import winsound
from threading import Thread



# Demo file (optional)
savedir         = '../_Archive_/'
video_in        = '../_Archive_/DEMO-01.avi'  # Use 0 for webcam, or path to file e.g.: '../_Archive_/DEMO-01.avi' 
video_out       = os.path.join(savedir,'DEMO01-OUT.avi')

# Save video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_out, fourcc, 10, (1440,960))


simulator = improc.PhospheneSimulator(intensity=10, phosphene_resolution=(50,50))



# preprocessing models
mode = 'SN' #Choose 'SN' or 'CED'
ced_filter = improc.CannyFilter(sigma=3,low=25,high=50)
sn_filter = improc.SharpNetFilter(checkpoint_path='./model/weights/final_checkpoint_NYU.pth',
                                device='cuda:0',smooth_output=1.5,threshold_b = 94, threshold_c=70)


cap = cv2.VideoCapture(video_in)
ret,frame = cap.read()
nFrames=0
while frame is not None:

    # Grab frame (zoom and center_crop)
    # frame  = improc.center_crop(frame,zoom=1.9)
    nFrames = nFrames+1
    # print(n_frames)

    # SharpNet predictions (resized output: 480 x 480)
    pred    = sn_filter(frame, resize=(480,480), store_raw = True)
    boundary = sn_filter.boundary
    normals = sn_filter.normals

    # Canny edge detection (resized input 480 x 480)
    frame = cv2.resize(frame, (480,480))
    canny = ced_filter(frame)


    # Phosphene simulation
    if mode == 'CED':
        phosphenes = simulator(canny)
    elif mode== 'SN':
        phosphenes = simulator(pred)

    # Convert to openCV 3-channel format
    canny =  cv2.cvtColor(canny,cv2.COLOR_GRAY2BGR)
    pred =  cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR)


    # Concatenate images
    up = np.concatenate([frame, normals, boundary],axis=1)
    down = np.concatenate([canny, pred, phosphenes],axis=1)
    total = np.concatenate([up,down],axis=0)

    # out.write(total)
    cv2.imshow('frame',total)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    ret,frame = cap.read()


cap.release()
out.release()
