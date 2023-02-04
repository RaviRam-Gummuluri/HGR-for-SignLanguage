import math
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time 

cam=cv2.VideoCapture(0)
det=HandDetector(maxHands=2)

BOXOFFSET=20
IMGSIZE=200

char=input('Enter the character : ')
folder = "Data/"+char
counter = 0

while True:
    succ, img = cam.read()
    hands, img = det.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        aspectratio = h/w

        croppedimg = img[y-BOXOFFSET:y+h+BOXOFFSET, x-BOXOFFSET:x+w+BOXOFFSET]

        whiteimg = np.ones((IMGSIZE,IMGSIZE,3),np.uint8)*255

        if aspectratio > 1:
            k = IMGSIZE / h
            wc = math.ceil(k*w)
            hc = IMGSIZE

        else:
            k = IMGSIZE /w
            hc = math.ceil(k*h)
            wc = IMGSIZE

        imgresize = cv2.resize(croppedimg, (wc, hc))
        sx = math.ceil((IMGSIZE-imgresize.shape[0])/2)
        sy = math.ceil((IMGSIZE-imgresize.shape[1])/2)
        
        whiteimg[sx:imgresize.shape[0]+sx,sy:imgresize.shape[1]+sy] = imgresize

        cv2.imshow("imgcrp",croppedimg)
        cv2.imshow("whiteimg",whiteimg)

    cv2.imshow("img",img)
    cv2.waitKey(1) #task is to handle out of bound box
    key = cv2.waitKey(1)
    if key==ord("s"):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',whiteimg)
        print(counter)


