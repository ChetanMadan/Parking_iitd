import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import sys
import time
sys.path.append('.')
from samples.coco import coco
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "data")

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


precision = 10
from datetime import datetime

def getCurrentClock():
    #return time.clock()
    return datetime.now()

start_time = time.time()
import pafy

start_time = time.time()



import cv2
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
image=cv2.imread("data/try.jpg")
results = model.detect([image], verbose=1,)
import json
with open('data.json','w') as outfile:
    i=0
    r = results[0]
    for class_id in r['class_ids']:
        print(class_names[class_id]) 
        data = {class_names[class_id] : {
        "bottom left x ":str(r['rois'][i][1]) ,
        "bottom left y " : str(r['rois'][i][2]) ,
        "top right x ":str(r['rois'][i][3]) ,
        "top right y" : str(r['rois'][i][0]) }  
        } 
        print("bottom left : "+str(r['rois'][i][1])+","+str(r['rois'][i][2])+"\n top right : " + str(r['rois'][i][3])+","+str(r['rois'][i][0]))
        i+=1
        json.dump(data,outfile)




dist=0
data_cars=[]
def find_dist(x1,y1,x2,y2):
    dist=pow(((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)),0.5)
    print(dist)
    return(dist)
a=0
b=0

def detect_spaces(image, data_cars,a,b):
    height, width=image.shape[:2]
    d1=int(data_cars[a]['car']['bottom left x '])
    d2=int(data_cars[a]['car']['bottom left y '])
    e1=int(data_cars[b]['car']['bottom left x '])
    e2=int(data_cars[b]['car']['bottom left y '])
    val=find_dist(d1,d2,e1,e2)
    final_cal=val/(540-d1)
    print(final_cal)
    return final_cal
    print(height,width)
i=0
for class_id in r['class_ids']:
    if class_names[class_id] in("car","truck","bicycle","bus"):
        data_cars.append({class_names[class_id] : {
        "bottom left x ":str(r['rois'][i][1]) ,
        "bottom left y " : str(r['rois'][i][2]) ,
        "top right x ":str(r['rois'][i][3]) ,
        "top right y" : str(r['rois'][i][0]) }  
        })
    i+=1

k=0
j=0
for k in range(len(data_cars)):
    for j in range(len(data_cars)):
        if k!=j:
            final_dist=detect_spaces(image,data_cars,k,j)
            if final_dist >- 0.575 and final_dist<-0.375:
                print("parking space is available between",k," and ",j)

final_dist=detect_spaces(image,data_cars,0,2)

if final_dist > -0.5 and final_dist<-0.375:
    print("parking space is available")



visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])

i=0

print(data_cars)