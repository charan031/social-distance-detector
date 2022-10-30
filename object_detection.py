import cv2
import numpy as np

thres = 0.45 # Threshold to detect object
nms_threshold = 0.2
cap = cv2.VideoCapture("C:\\Users\\chara\\Downloads\\TSF_ComputerVision_Internship-main\\Object_Detection\\pedestrians.mp4")
# cap.set(3,1280)
# cap.set(4,720)
# cap.set(10,150)

classNames= []
classFile = "C:\\Users\\chara\\Downloads\\TSF_ComputerVision_Internship-main\\Object_Detection\\coco.names"
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#print(classNames)
configPath = "C:\\Users\\chara\\Downloads\\TSF_ComputerVision_Internship-main\\Object_Detection\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "C:\\Users\\chara\\Downloads\\TSF_ComputerVision_Internship-main\\Object_Detection\\frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    img=cv2.resize(img,(0,0),None,0.7,0.7)
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(type(confs[0]))
    #print(confs)

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    #print(indices)

    for i in indices:
        #i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
        #cv2.putText(img,(box[0]+10,box[1]+30),
                    #cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("Output",img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
