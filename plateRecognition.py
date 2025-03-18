import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

path = 'placas.jpg'

# Read the image in grayscale
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply binary thresholding to emphasize edges
_, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)

# Sobel in x
sobel_x = cv2.Sobel(thresholded, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)

# Sobel in y
sobel_y = cv2.Sobel(thresholded, cv2.CV_64F, 0, 1, ksize=3)
sobel_y = cv2.convertScaleAbs(sobel_y)

# Sobel combined
sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# Use Tesseract to extract text from the Sobel combined image
# Convert the image to a format suitable for Tesseract (uint8)
sobel_combined_uint8 = cv2.convertScaleAbs(sobel_combined)
text = pytesseract.image_to_string(sobel_combined_uint8, config='--psm 8')  # Page segmentation mode 8 to treat the image as a single word

# Print the extracted text
print("Extracted Text:")
print(text)

# Display the result
plt.figure(figsize=(12, 8))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Final")
plt.imshow(sobel_combined, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()