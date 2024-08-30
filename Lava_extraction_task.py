import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow


image = cv2.imread('lava2.jpg')
gray_image = cv2.imread('lava21.jpg', 0)  # Grayscale image for edge detection
height, width, channels = image.shape
# Split the image into its channels
b, g, r = cv2.split(image)

# Applying Gaussian blur to the red channel
blurred_r = cv2.GaussianBlur(r, (5, 5), 0)

# Apply Otsu's thresholding to the red channel
ret, thresh_r = cv2.threshold(blurred_r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Combine the thresholded red channel with the original green and blue channels
thresh = cv2.merge((thresh_r, g, b))

# Display the original and thresholded images
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
plt.title('Otsu Thresholding on Red Channel'), plt.xticks([]), plt.yticks([])
plt.show()

# Apply GaussianBlur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)

# Find edges using the Canny edge detector
edges = cv2.Canny(blurred, 30, 100)  # You can adjust these thresholds according to your needs
edge_count = np.sum(edges != 0)
ar=height*width
segmented_areas = np.uint8(thresh_r)
segmented_areas_shape = segmented_areas.shape
filtered_segmented_areas = np.zeros(segmented_areas_shape, dtype=float)
if(edge_count/ar > 0.002) :
    for y in range(segmented_areas.shape[0]):
        for x in range(segmented_areas.shape[1]):
            if segmented_areas[y, x] != 0:  # Check if the pixel belongs to a segmented area
                start_y = max(0, y - 10)
                end_y = min(segmented_areas.shape[0], y + 11)
                start_x = max(0, x - 10)
                end_x = min(segmented_areas.shape[1], x + 11)
                nearby_region = edges[start_y:end_y, start_x:end_x] * segmented_areas[start_y:end_y, start_x:end_x]
                if np.sum(nearby_region) >= 5:
                    filtered_segmented_areas[y, x] = 255
else :
    for y in range(segmented_areas.shape[0]):
        for x in range(segmented_areas.shape[1]):
            if segmented_areas[y, x] != 0:  # Check if the pixel belongs to a segmented area
                start_y = max(0, y - 10)
                end_y = min(segmented_areas.shape[0], y + 11)
                start_x = max(0, x - 10)
                end_x = min(segmented_areas.shape[1], x + 11)
                nearby_region = edges[start_y:end_y, start_x:end_x] * segmented_areas[start_y:end_y, start_x:end_x]
                if np.sum(nearby_region) >= 0:
                    filtered_segmented_areas[y, x] = 255
filtered_segmented_areas = filtered_segmented_areas.astype(np.uint8)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filtered_segmented_areas, connectivity=8)

min_segment_area = 50  # Define the minimum area for segmented regions
for label in range(1, num_labels):
    if stats[label, cv2.CC_STAT_AREA] < min_segment_area:
        filtered_segmented_areas[labels == label] = 0

# Include non-segmented regions that lie within a segmented area
for y in range(filtered_segmented_areas.shape[0]):
    for x in range(filtered_segmented_areas.shape[1]):
        if filtered_segmented_areas[y, x] == 0:  # Check if the pixel is non-segmented
            start_y = max(0, y - 10)
            end_y = min(filtered_segmented_areas.shape[0], y + 11)
            start_x = max(0, x - 10)
            end_x = min(filtered_segmented_areas.shape[1], x + 11)
            nearby_region = filtered_segmented_areas[start_y:end_y, start_x:end_x]
            nearby_segmented_pixels = np.sum(nearby_region == 255)
            if nearby_segmented_pixels > 0.5 * nearby_region.size:
                filtered_segmented_areas[y, x] = 255

# Find contours and draw the largest contour
threshold_ratio = 0.06
contours, _ = cv2.findContours(filtered_segmented_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) > 1:
    max_contour = max(contours, key=cv2.contourArea)
    final_segmented_areas = np.zeros_like(filtered_segmented_areas)
    cv2.drawContours(final_segmented_areas, [max_contour], -1, 255, thickness=cv2.FILLED)
    cv2_imshow(final_segmented_areas)
    # Invert the filtered segmented areas
    inverted_filtered_segmented_areas = cv2.bitwise_not(final_segmented_areas)

    # Find contours of the non-segmented areas
    contours_non_segmented, _ = cv2.findContours(inverted_filtered_segmented_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours of non-segmented areas
    image_contours = np.copy(image)
    cv2.drawContours(image_contours, contours_non_segmented, -1, (0, 255, 0), 2)

    # Display the image with contours
    cv2_imshow(image_contours)
    for contour in contours_non_segmented:
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if area > 0:
            perimeter_area_ratio = perimeter / area
            if perimeter_area_ratio > threshold_ratio:
                cv2.drawContours(final_segmented_areas, [contour], -1, 255, thickness=cv2.FILLED)

    cv2_imshow(final_segmented_areas)

else:
    cv2_imshow(filtered_segmented_areas)
    # Invert the filtered segmented areas
    inverted_filtered_segmented_areas = cv2.bitwise_not(filtered_segmented_areas)

    # Find contours of the non-segmented areas
    contours_non_segmented, _ = cv2.findContours(inverted_filtered_segmented_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours of non-segmented areas
    image_contours = np.copy(image)
    cv2.drawContours(image_contours, contours_non_segmented, -1, (0, 255, 0), 2)

    # Display the image with contours
    cv2_imshow(image_contours)
    for contour in contours_non_segmented:
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if area > 0:
            perimeter_area_ratio = perimeter / area
            if perimeter_area_ratio > threshold_ratio:
                cv2.drawContours(filtered_segmented_areas, [contour], -1, 255, thickness=cv2.FILLED)

    cv2_imshow(filtered_segmented_areas)
