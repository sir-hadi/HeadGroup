import os
import time
import argparse
import cv2
import numpy as np
import sys
import importlib.util
from scipy.spatial import distance as dist
import random

import time
import random
import paho.mqtt.client as paho

def on_connect(client, userdata, flags, rc):
    print('CONNECTED received with code %d.' % (rc))

broker = 'broker.emqx.io'
port = 1883
topic_header = "/headGroup/"
client_id = f'head-group-{random.randint(0, 1000)}'

client = paho.Client(client_id)
client.on_connect = on_connect
client.connect(broker, port)

# Connect to MQTT Broker
# while( not client.is_connected() ):
#     print('.', end='')
#     time.sleep(1)

def publish(client,topic,msg):
    result = client.publish(topic, str(msg))
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Send `{msg}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")


def masuk_region_mana(C):
    x, y = C[0], C[1]
    if x<960 and y<540:
        return 1
    elif x>960 and y<540:
        return 2
    elif x<960 and y>540:
        return 3
    elif x>960 and y>540:
        return 4
    else:
        return 4

    


# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

import pandas

# Read data points
my_data = pandas.read_csv('data.csv')

MODEL_NAME = 'model'
INPUT_PATH = 0 # 0 means we use the built-in camera or the default one
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
min_conf_threshold = 0.5

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
frame_rate_list = []
freq = cv2.getTickFrequency()

# open video camera
video = cv2.VideoCapture(INPUT_PATH)
video.set(3, 1920)
video.set(4, 1080)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps_video = int(video.get(cv2.CAP_PROP_FPS))

while(video.isOpened()):
    jumlah = dict.fromkeys(my_data.values[:,0],0)
    print("emptying jumlah")

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if(INPUT_PATH == 0):
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    if not ret:
        print('Output video is generated in the directory!')
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    start = time.perf_counter()
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

    inference_time = time.perf_counter() - start
    
    p_boxes = []
    p_centroids = []
    p_scores = []
    res = []
    countF = 0
    prevCenter = [0,0] ###holds prev center point

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)) and labels[int(classes[i])] == "person":
            
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            area = (xmax - xmin) * (ymax - ymin)

            centerX = (xmin+xmax)/2
            centerY = (ymin+ymax)/2
            lebihanX = prevCenter[0]+20
            lebihanY = prevCenter[1]+20
            kuranganX = prevCenter[0]-20
            kuranganY = prevCenter[1]-20
            if(((centerX<lebihanX) and (centerX>kuranganX) and (centerY<lebihanY) and (centerY>kuranganY)) or ((prevCenter[0]==0) and (prevCenter[1]==0)) ):
                p_boxes.append([xmin, ymin, xmax, ymax])
                p_centroids.append((centerX, centerY))
                p_scores.append(float(scores[i]))
                r = (p_scores, p_boxes, p_centroids)
                res.append(r)

    

        
    # di jadiin tuple
    for c in p_centroids:
        regs = masuk_region_mana((c[0], c[1]))
        # misal dia region 1
        # ngebandingin pada region 1 aja
        point = my_data.loc[my_data.reg == regs]
        all_dist = []
        for i in range(point.shape[0]):
            d = dist.euclidean( (point.iloc[int(i),1],point.iloc[int(i),2]) , (c[0], c[1]) )
            t = (point.iloc[int(i),0],(point.iloc[int(i),1],point.iloc[int(i),2]),(c[0],c[1]),d)
            all_dist.append(t)

        all_dist.sort(key=lambda x: x[3])

        update = jumlah[all_dist[0][0]] + 1 
        jumlah[all_dist[0][0]] = update


    # publish data
    for key, value in jumlah.items():
        publish(client,topic_header+key,value)

    # Draw Boinding Box
    for (i, (prob, bbox, centroid)) in enumerate(res):
        xmin, ymin, xmax, ymax = bbox[i]
        (cX, cY) = centroid[i]
        color = (0, 255, 0)               
        
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 4)
        cv2.circle(frame, (int(cX), int(cY)), 5, color, 1)
    
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    print('FPS: {0:.2f}'.format(frame_rate_calc))
    frame_rate_list.append(frame_rate_calc)

    # draw point
    color = (255, 0, 0)
    for i in range(my_data.shape[0]):
        cv2.circle(frame, (int(my_data.iloc[int(i),1]), int(my_data.iloc[int(i),2])), 5, color, 1)

    name = "frameGroup.jpg"
    cv2.imwrite(name, frame)

    print("Jumlah orang per point : ")
    print(jumlah)
    
    print('-------------------------------------')
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Social Distancing', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
# Clean up
if (len(frame_rate_list) != 0):
    print('AVG FPS :',(sum(frame_rate_list)/len(frame_rate_list)) )
video.release()
cv2.destroyAllWindows()