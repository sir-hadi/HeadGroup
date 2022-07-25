import os
import time
import argparse
import cv2
import numpy as np
import sys
import importlib.util
from scipy.spatial import distance as dist

import pandas
my_data = pandas.read_csv('data.csv').values



countF = 0

"""
    Parse command line arguments.
    
    :return: command line arguments
"""
parser = argparse.ArgumentParser()

parser.add_argument("-m", "--modeldir", required=True, type=str,
                        help="Folder path to the .tflite file.")

parser.add_argument("-g", "--graph", type=str,
                    help="Name of the .tflite file, if different than detect.tflite.",
                    default='detect.tflite')

parser.add_argument("-l", "--labels", type=str,
                    help="Name of the labelmap file, if different than labelmap.txt.",
                    default='labelmap.txt')

parser.add_argument("-i", "--input", required=True, type=str,
                    help="Path to image or video file or CAM.")

parser.add_argument("-pt", "--threshold", help='Probability threshold for detection filtering',
                    default=0.5)

parser.add_argument('--edgetpu',
                    help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
INPUT_NAME = args.input
min_conf_threshold = float(args.threshold)
use_TPU = False

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'   

# Get path to current working directory
CWD_PATH = os.getcwd()
#print(CWD_PATH)

im_flag = False

if(INPUT_NAME == 'CAM'):
    INPUT_PATH = 0
    
elif(INPUT_NAME.endswith('.jpg') or INPUT_NAME.endswith('.bmp')):
    #INPUT_PATH = os.path.join(CWD_PATH,INPUT_NAME)
    INPUT_PATH = CWD_PATH + INPUT_NAME
    im_flag = True
    
else:# Path to video file
    INPUT_PATH = os.path.join(CWD_PATH,INPUT_NAME)
    INPUT_PATH = CWD_PATH + INPUT_NAME
    im_flag = False
    
print(INPUT_PATH)
base = os.path.basename(INPUT_NAME)
filename = os.path.splitext(base)[0]
#print(filename)

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
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    #print(PATH_TO_CKPT)
else:
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


print('video')
# Open video file
print(INPUT_PATH)
video = cv2.VideoCapture(INPUT_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps_video = int(video.get(cv2.CAP_PROP_FPS))
print(imH, imW)
writefile = "out-" + filename + ".mp4"

while(video.isOpened()):
    start_time = time.time()

    jumlah = dict.fromkeys(my_data[:,0],0)

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
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0))and labels[int(classes[i])] == "person":
        #if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0))and labels[int(classes[i])] == "person":
            print(classes[i])
            print("Label:", end=" ")
            print(labels[int(classes[i])], end=", ")
            print("Confidence:", end=" ")
            print(scores[i])
            
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            print(xmax - xmin)
            print(ymax - ymin)
            area = (xmax - xmin) * (ymax - ymin)
            print("Area:", end=" ")
            print(area)
            
            """
            if(INPUT_PATH != 'CAM' and area > 90000):
                break
            """
            
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

                all_dist = []
                # di jadiin tuple
                for i in range(my_data.shape[0]):
                    d = dist.euclidean( (my_data[int(i),1],my_data[int(i),2]) , (centerX, centerY) )
                    t = (my_data[int(i),1],(my_data[int(i),1],my_data[int(i),2]),(my_data[0,1],my_data[0,2]),d)
                    all_dist.append(t)
                    print(t)

                all_dist.sort(key=lambda x: x[3])

                jumlah[all_dist[0][0]] =+ 1 

     
    for (i, (prob, bbox, centroid)) in enumerate(res):
        print(bbox)
        print(prob)
        print(centroid)
        
        xmin, ymin, xmax, ymax = bbox[i]
        (cX, cY) = centroid[i]
        color = (0, 255, 0)               
        
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 4)
        cv2.circle(frame, (int(cX), int(cY)), 5, color, 1)
            
        # Draw label
        print('%.2f ms' % (inference_time * 1000))

    for key, value in jumlah

    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    print('FPS: {0:.2f}'.format(frame_rate_calc))
    frame_rate_list.append(frame_rate_calc)

    # draw point
    color = (255, 0, 0)
    for i in range(my_data.shape[0]):
        cv2.circle(frame, (int(my_data[int(i),1]), int(my_data[int(i),2])), 5, color, 1)
    
    countF=countF+1
    name = "ilhamF/frame%d.jpg"%countF
    cv2.imwrite(name, frame)

    print("Jumlah orang per point : ")
    print(jumlah)
    
    print('-------------------------------------')
    # All the results have been drawn on the frame, so it's time to display it.
    #frame = cv2.resize(frame, (int(imW/4), int(imH/4)), interpolation = cv2.INTER_AREA)
    cv2.imshow('Social Distancing', frame)
    

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

    time.sleep(5.0 - time.time() + start_time) # Sleep for 1 second minus elapsed time

# Clean up
if (len(frame_rate_list) != 0):
    print('AVG FPS :',(sum(frame_rate_list)/len(frame_rate_list)) )
video.release()
cv2.destroyAllWindows()
