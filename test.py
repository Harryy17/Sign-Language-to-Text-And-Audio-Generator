import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from cvzone.ClassificationModule import Classifier
import pyttsx3
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=2)

# Initialize the classifier with the model and label paths
classifier = Classifier("new_model\keras_model.h5", "new_model\labels.txt")

# Parameters for preprocessing
offset = 20
imgSize = 300

# Define class labels
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Variables for controlling audio feedback
last_prediction = None
last_prediction_time = 0
prediction_threshold = 0.8  # Confidence threshold
cooldown_time = 2.0  # Time to wait before repeating the same prediction

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    # Detect hands in the frame
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure crop coordinates are within image bounds
        y1 = max(y - offset, 0)
        y2 = min(y + h + offset, img.shape[0])
        x1 = max(x - offset, 0)
        x2 = min(x + w + offset, img.shape[1])

        # Crop the hand region from the image
        imgCrop = img[y1:y2, x1:x2]

        # Get the shape of the cropped image
        imgCropShape = imgCrop.shape

        # Calculate aspect ratio of the bounding box
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = max(int(w * k), 1)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = max(int(h * k), 1)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Make a prediction
        prediction, index = classifier.getPrediction(imgWhite)
        current_time = time.time()
        
        # Check if prediction meets confidence threshold
        if prediction[index] > prediction_threshold:
            predicted_letter = labels[index]
            
            # Handle audio feedback with cooldown
            if (predicted_letter != last_prediction or 
                current_time - last_prediction_time > cooldown_time):
                # Update the display
                cv2.putText(img, f"{predicted_letter}: {int(prediction[index] * 100)}%", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Provide audio feedback
                engine.say(f"Letter {predicted_letter}")
                engine.runAndWait()
                
                # Update tracking variables
                last_prediction = predicted_letter
                last_prediction_time = current_time
            else:
                # Display prediction without audio
                cv2.putText(img, f"{predicted_letter}: {int(prediction[index] * 100)}%", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the cropped and resized images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the webcam feed
    cv2.imshow("Image", img)

    # Break the loop on 'q' key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Cleanup
engine.stop()
cap.release()
cv2.destroyAllWindows()