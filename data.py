import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgsize = 300
folder = "images\SREEHARI "
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    imgWhite = None  # Initialize imgWhite outside the if block
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Create white background image
        imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
        
        # Ensure crop coordinates are within image bounds
        y1 = max(y - offset, 0)
        y2 = min(y + h + offset, img.shape[0])
        x1 = max(x - offset, 0)
        x2 = min(x + w + offset, img.shape[1])
        
        # Crop the hand region
        imgCrop = img[y1:y2, x1:x2]
        
        # Get the shape of cropped image
        imgCropShape = imgCrop.shape
        
        # Calculate aspect ratio
        aspectRatio = h/w
        
        if aspectRatio > 1:
            # Adjust width when height is bigger
            k = imgsize/h
            wCal = max(int(w * k), 1)
            imgResize = cv2.resize(imgCrop, (wCal, imgsize))
            imgResizeShape = imgResize.shape
            wGap = int((imgsize - wCal)/2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            # Adjust height when width is bigger
            k = imgsize/w
            hCal = max(int(h * k), 1)
            imgResize = cv2.resize(imgCrop, (imgsize, hCal))
            imgResizeShape = imgResize.shape
            hGap = int((imgsize - hCal)/2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        # Only save if a hand was detected and imgWhite was created
        if imgWhite is not None:
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(counter)
        else:
            print("No hand detected - cannot save image")
            
cap.release()
cv2.destroyAllWindows()