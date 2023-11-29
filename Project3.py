import cv2
import numpy as np

# Load the motherboard image
image_path = 'data/motherboard_image.JPEG'
original_image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)


# Apply thresholding to segment the image based on pixel intensity
_, thresholded_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

# Perform edge detection using the Canny edge detector
#edges = cv2.Canny(thresholded_image, 30, 100)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours based on area
filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]

# Create an empty mask to draw the contours
mask = np.zeros_like(original_image)

# Draw the filtered contours on the mask
cv2.drawContours(mask, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

# Use bitwise AND to extract the PCB from the original image
extracted_image = cv2.bitwise_and(original_image, mask)


'''
#Corner Detection only:

# Detect corners using the Harris corner detector
#corners = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)
corners = cv2.Canny(gray_image, 30, 100)


# Threshold the corner response to get strong corners
threshold = 0.01 * corners.max()
corner_mask = np.zeros_like(gray_image)
corner_mask[corners > threshold] = 255

extracted_image = cv2.bitwise_and(original_image, original_image, mask=corner_mask)
'''




# Save the extracted image
cv2.imwrite('data/extracted_image.JPEG', extracted_image)

# Display the original and extracted images
cv2.imshow('Original Image', original_image)
cv2.imshow('Extracted Image', extracted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
