# %%
#### 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'seismic/s01.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Edge Detection using Canny
edges = cv2.Canny(image, threshold1=100, threshold2=200)

# Detect line segments with HoughLinesP
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

# Filter for significant long edges by setting a minimum line length
min_line_length = 100  # Set your minimum line length here

# Draw the filtered long lines on a copy of the original image for visualization
image_with_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if line_length > min_line_length:
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the original image with significant long edges overlaid
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))
plt.title('Significant Long Edges')
plt.axis('off')
plt.show()


# %%
