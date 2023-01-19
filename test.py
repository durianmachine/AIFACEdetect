import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os
import numpy as np

#displays unavaliable devices
if torch.backends.mps.is_available() == False:
 print('Metal GPU unavaliable.')
if torch.cuda.is_available()== False:
    print("CUDA device unavaliable")
    
#displays avaliable devices: 
if torch.cuda.is_available(): 
 dev = "cuda:0" 
 print('CUDA Avalible')
elif torch.backends.mps.is_available() == True:
 dev= "mps:0"
 print('Metal Avaliable')
else: #if metal and cuda false, pick device.
 deviceselect = input('No dedicated computing. Select device to be used (cpu, cuda, mps, etc): ').lower()
 dev = deviceselect

#temporary cause TorchVision does not support MPS compute
if torch.backends.mps.is_available() == True:
 dev = input('MPS is currently unsupported. Select device to be used (cpu, cuda, etc): ').lower()

time.sleep(0.3)
print('Using: ' + dev + ' compute') #Prints out computing device


mtcnn0 = MTCNN(image_size = 140, keep_all =False, min_face_size= 35, device =dev)
mtcnn1 = MTCNN(image_size = 140, keep_all= True, min_face_size= 35, device =dev)
resnet = InceptionResnetV1(pretrained='vggface2').eval() #pretrained face detection model from keras Library

#reading files
dataset = datasets.ImageFolder('images')
classes = {z:a for a,z in dataset.class_to_idx.items()} #name of ppl from folder name
print(classes)

def collate_fn(x):
    return x[0]

name_list = []
embedding_list = []

for img, idx in DataLoader(dataset, collate_fn=collate_fn): #Runs the face detection code if a face is detected via the vggface model
    face, prob = mtcnn0(img, return_prob=True) 


torch.save([embedding_list, name_list], 'test.pt')
ld = torch.load('test.pt') 
embedding_list = ld[0] 
name_list = ld[1] 

cap = cv2.VideoCapture(0) 
cap_fps = cap.get(cv2.CAP_PROP_FPS)

#LOGS THE ATTENDANCE
def logs():
    print(name + ' detected at: ' +  time.strftime("%H:%M:%S", time.localtime()) + ' accuracy(%): '+str(int(min(dist_list)*100)))

while True:
    grep, frame = cap.read()
        
    image = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn1(image, return_prob=True) 
    
    if img_cropped_list is not None:
        boxes, _ = mtcnn1.detect(image)
                
        for i, prob in enumerate(prob_list):
            if prob>0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                dist_list = [] # list of matched distances, minimum distance is used to identify the person

                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)
                    
                name = name_list[dist_list.index(min(dist_list))]
                if min(dist_list)<0.90: logs()

    if dev == 'cuda:0':
        devprint = 'NVIDIA COMPUTE' #Computing device string
    elif dev == 'mps:0':
        devprint = "METAL COMPUTE"
    else:
        devprint = "CPU COMPUTE"
    
    if 12 > int(time.strftime("%H", time.localtime())) > 0: message = 'Good Afternoon'
    else: message = 'Good Evening'

    #shows sign, fps, computing device via devprint
    currentdev = cv2.putText(frame, (devprint + ' FPS:' + str(cap_fps)), (25,25), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0),1, cv2.LINE_AA)
    cv2.putText(frame, message, (100,100), cv2.FONT_HERSHEY_DUPLEX, 3, (0,0,0),8, cv2.LINE_AA)
    cv2.imshow("WELCOME", frame)

    k = cv2.waitKey(1)
    if k%256==27: # press escape key to close
        break
        
cap.release()
cv2.destroyAllWindows()