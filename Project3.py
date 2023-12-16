import cv2
import numpy as np

# Load the motherboard image
image_path = 'data/motherboard_image.JPEG'
original_image = cv2.imread(image_path)

'''
Part 1: Object Masking
'''

'''
DELETE BACKGROUND
'''

# Convert the image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)


# Apply thresholding to segment the image based on pixel intensity
_, thresholded_image = cv2.threshold(gray_image, 90, 255, cv2.THRESH_BINARY) #max was 128
cv2.imwrite('data/threshold.JPEG', thresholded_image)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < 100000000]

# Create an empty mask to draw the contours
mask = np.zeros_like(gray_image)

# Draw the filtered contours on the mask
cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

# Use bitwise AND to extract the PCB from the original image
extracted_image = cv2.bitwise_and(original_image, original_image, mask = mask)


'''
DELETE DESK
'''

# Convert the image to grayscale
gray_image = cv2.cvtColor(extracted_image, cv2.COLOR_BGR2GRAY)


# invert grayscale
thresholded_image = cv2.bitwise_not(thresholded_image)
cv2.imwrite('data/invert.JPEG', thresholded_image)


# Find contours in the edge-detected image
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < 100000000]

# Create an empty mask to draw the contours
mask = np.zeros_like(gray_image)

# Draw the filtered contours on the mask
cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

# Use bitwise AND to extract the PCB from the first extracted image
final_image = cv2.bitwise_and(extracted_image, original_image, mask = mask)

# Save the final image of ONLY the PCB
cv2.imwrite('data/final_image.JPEG', final_image)





'''
Part 2: YOLO Training
'''

from ultralytics import YOLO
import os
from IPython.display import display, Image
from IPython import display
display.clear_output()

'''
from roboflow import Roboflow
project = Roboflow.workspace("spudzy-y0jgr").project("Project3")
dataset = project.version(1).download("yolov8")
'''

# Load a model
model = YOLO('yolov8n.pt')  #load a pretrained model


#results = model(source='data.yaml', show = True, conf=0.4, save=True)



# Use the model
model.train(data='data.yaml', epochs=2, batch = 5, imgsz = 900)  # train the model

print('testing')
metrics = model.val()  # evaluate model performance on the validation set
results = model("data/evaluation/arduno.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format

print('sucess!')



