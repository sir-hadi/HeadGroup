import numpy as np
import math
import cv2
import time
from scipy.spatial import distance as dist

import pandas
my_data = pandas.read_csv('data.csv').values
print(my_data)

jumlah = dict.fromkeys(my_data[:,0],0)

# coor = {}

# for i in range(my_data.shape[0]):
#     # print(i,(my_data[int(i),1],my_data[int(i),2]))
    
#     coor[my_data[int(i),0]] = (my_data[int(i),1],my_data[int(i),2])

# all_dist = []
# # di jadiin tuple
# for i in range(1,my_data.shape[0]):
#     d = dist.euclidean( (my_data[int(i),1],my_data[int(i),2]) , (my_data[0,1],my_data[0,2]) )
#     t = (my_data[int(i),0],(my_data[int(i),1],my_data[int(i),2]),(my_data[0,1],my_data[0,2]),d)
#     all_dist.append(t)
#     print(t)

# all_dist.sort(key=lambda x: x[3])


# print('shortest')    
# print(all_dist[0])
# print("id : ", all_dist[0][0])

# jumlah[all_dist[0][0]] =+ 1 

# print(jumlah)


# print("-"*5)
# a = math.sqrt((my_data[2,1] - my_data[0,1])**2 + (my_data[2,2] - my_data[0,2])**2)
# print(a)

# print(coor)

# print(my_ids)

# for i in my_ids:
#     print(my_ids[i])


cap = cv2.VideoCapture(0)
imW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
imH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(imW,imH)


font = cv2.FONT_HERSHEY_SIMPLEX
offset = 0
numberOfId = my_data.shape[0]

while(True):
    # Capture frame-by-frame
    # start_time = time.time()
    ret, frame = cap.read()

    offset = 100
    cv2.putText(frame, 'Number of people detected', (imW-500,offset), font, 1, (0, 255, 0),3)
    for i in range(numberOfId):
        # offset += int(imH / numberOfId) - 10
        offset += imH - 1000
        # print((imW-50,offset))
        cv2.putText(frame, f'Point {my_data[int(i),0]}: 0 People', (imW-500,offset), font, 1, (0, 255, 0),3)

    # Display the resulting frame
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # time.sleep(30.0 - time.time() + start_time) # Sleep for 1 second minus elapsed time

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()