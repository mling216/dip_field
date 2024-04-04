# %%
#### 
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters

# Function to calculate the dip angle
def calculate_dip_angle(window_gx, window_gy):
    covariance_matrix = np.cov(np.vstack([window_gx, window_gy]))
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    principal_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
    angle = np.arctan2(principal_eigenvector[1], principal_eigenvector[0])
    return np.degrees(angle)

# Load your image (example)
image_path = 'seismic/s03.jpg'
image = io.imread(image_path, as_gray=True)
print(image.shape)

# Display the image
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()



# %%
#### Calculate gradients
gx = filters.sobel_h(image)
gy = filters.sobel_v(image)
gradient_magnitude = np.sqrt(gx**2 + gy**2)

# Define a threshold for gradient magnitude
magnitude_threshold = np.percentile(gradient_magnitude, 75)  # Example: keep top 25% strongest signals

# Define your ROI (for the entire image here, but can be adjusted)
roi_start_x, roi_start_y = 0, 0
roi_end_x, roi_end_y = image.shape[1], image.shape[0]

# Window size for calculating dip angle
window_size = 5  # Must be odd
half_window = window_size // 2

# Initialize the dip field map
dip_field_map = np.full(image.shape, np.nan)  # Initialize with NaNs

# Initialize an empty list for points
significant_points = []

angle_threshold = 30

res = []
# Iterate over each pixel in the ROI and calculate dip angle if above threshold
for y in range(roi_start_y + half_window, roi_end_y - half_window):
    for x in range(roi_start_x + half_window, roi_end_x - half_window):
        # Only proceed if gradient magnitude exceeds the threshold
        if gradient_magnitude[y, x] > magnitude_threshold:
            window_gx = gx[y-half_window:y+half_window+1, x-half_window:x+half_window+1].flatten()
            window_gy = gy[y-half_window:y+half_window+1, x-half_window:x+half_window+1].flatten()
            dip_angle = calculate_dip_angle(window_gx, window_gy)
            # if (dip_angle > angle_threshold) or (dip_angle < (-1*angle_threshold)):
            if dip_angle < (-1*angle_threshold):
                dip_field_map[y, x] = dip_angle
                significant_points.append((x, y))
            else:
                dip_field_map[y, x] = np.nan  # Use NaN or another indicator for angles below the threshold

# Visualization of the dip field map with strong signals only
plt.imshow(dip_field_map, cmap='hsv', interpolation='nearest')
plt.colorbar(label='Dip Angle (degrees)')
plt.title('Dip Field Map with Strong Signals')
# plt.axis('off')
plt.show()



# %%
#### line detection
from sklearn.cluster import DBSCAN
import numpy.linalg as la

significant_points = np.array(significant_points)

# fix the upside down issue
image_height = image.shape[0]
inverted_points = np.copy(significant_points)
inverted_points[:, 1] = image_height - significant_points[:, 1]

# DBSCAN clustering
clustering = DBSCAN(eps=3, min_samples=5).fit(inverted_points)
labels = clustering.labels_

# Plot clusters
plt.scatter(inverted_points[:, 0], inverted_points[:, 1], c=labels, 
            cmap='viridis', s=5)



# %%
#### Fit and plot lines or polynomials for each cluster

# Define a minimum length for lines to be plotted
min_line_length = 10  # Adjust this threshold as needed

for label in set(labels):
    if label != -1:  # -1 is noise in DBSCAN
        cluster_points = significant_points[labels == label]
        if len(cluster_points) > 1:  # Need at least two points to fit a line
            # Calculate the pairwise distances between points in the cluster
            distances = la.norm(cluster_points[:, None] - cluster_points, axis=-1)
            # Find the maximum distance
            max_distance = np.max(distances)
            # Proceed if the longest line in the cluster exceeds the threshold
            if max_distance > min_line_length:
                # Fit a line (1st-degree polynomial)
                coef = np.polyfit(cluster_points[:, 0], cluster_points[:, 1], 1)
                poly = np.poly1d(coef)
                # Determine the end points of the line to plot
                x_vals = np.array([min(cluster_points[:, 0]), max(cluster_points[:, 0])])
                plt.plot(x_vals, poly(x_vals), color='blue')

plt.imshow(image, cmap='gray')
plt.show()



# %%
