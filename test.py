import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os

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
print('Using: ' + dev + ' compute') #Prints out computing device for MTCNN


mtcnn0 = MTCNN(image_size=240, keep_all=False, min_face_size=35, device =dev) # initializing the network while keeping keep_all = False
mtcnn1 = MTCNN(image_size=240, keep_all=True, min_face_size=35, device =dev) # initializing the network while keeping keep_all = True
resnet = InceptionResnetV1(pretrained='vggface2').eval() 

#reading files
dataset = datasets.ImageFolder('images') # photos folder path 
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} #name of ppl from folder name

def collate_fn(x):
    return x[0]

name_list = [] #correspond names to cropped photos
embedding_list = [] #conversion of cropped photos to matrix (vector) via pretrained facenet
load = DataLoader(dataset, collate_fn=collate_fn)

for img, idx in load:
    face, prob = mtcnn0(img, return_prob=True) 
    if face is not None and prob>0.9:
        emb = resnet(face.unsqueeze(0)) 
        embedding_list.append(emb.detach()) 
        name_list.append(idx_to_class[idx])        

data = [embedding_list, name_list] 
torch.save(data, 'data.pt') # saving data.pt file

load_data = torch.load('data.pt') 
embedding_list = load_data[0] 
name_list = load_data[1] 

cap = cv2.VideoCapture(0) 
cap_fps = cap.get(cv2.CAP_PROP_FPS)

#LOGS THE ATTENDANCE
def logs():
    print(name + ' detected at: ' +  time.strftime("%H:%M:%S", time.localtime()) + ' accuracy(%): '+str(int(min_dist*100)))

while True:
    grep, frame = cap.read()
    if not grep:
        break

    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn1(img, return_prob=True) 

    if img_cropped_list is not None:
        boxes, _ = mtcnn1.detect(img)
        for y, prob in enumerate(prob_list):
            if prob>0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                dist_list = [] #list of matched distances, minimum distance is used to identify the person
                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)
                min_dist = min(dist_list) #min detect value
                min_dist_idx = dist_list.index(min_dist) #index names
                name = name_list[min_dist_idx] #correspond the name to index file
                box = boxes[y] 
                if min_dist<0.90: #If acccuracy mets a threshold, it displays as a name
                    frame = cv2.putText(frame, name+' '+str(int(round(min_dist*100, 1)))+"%", (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_DUPLEX, 1, (100,100,255),2, cv2.LINE_AA)
                    logs()
                   
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
