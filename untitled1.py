import cv2
import numpy as np

# Load the motherboard image
image_path = 'data/motherboard_image.JPEG'
original_image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to segment the image
_, thresholded_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours based on area
min_contour_area = 10000
max_contour_area = 234000

for i in range(2):
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < max_contour_area]

# Create an empty mask
mask = np.zeros_like(gray_image)

# Draw the filtered contours on the mask
cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

# Use bitwise_and to extract the PCB from the background
extracted_image = cv2.bitwise_and(original_image, original_image, mask=mask)


cv2.imwrite('data/test.JPEG', extracted_image)

# Display the original image and the extracted image
#cv2.imshow("Original Image", original_image)
cv2.imshow("Extracted Image", extracted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#model.train