import cv2
import numpy as np
from google.colab.patches import cv2_imshow
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY)
    inverted_thresh = cv2.bitwise_not(thresh)
    return image, gray, inverted_thresh

def find_largest_contour(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = contours[0]
    max_area = cv2.contourArea(max_contour)

    for contour in contours[1:]:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    return max_contour

def get_bounding_box(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h

def find_connected_components(binary_image):
    _, markers = cv2.connectedComponents(binary_image)
    return markers


def filter_small_components(component_contours):
    return [contour for contour in component_contours if cv2.contourArea(contour) >= 100]

def calculate_x_coordinates(cropped_inverted_thresh):
    x1, x2 = cropped_inverted_thresh.shape[1], 0
    neg=-1
    pos=0
    for i in range(cropped_inverted_thresh.shape[0] - 1, neg, neg):
        row = cropped_inverted_thresh[i, :]
        indices = np.where(row != pos)[pos]

        if indices.size > pos:
            if x1 > indices[pos]:
               x1=indices[pos]
            if x2 < indices[neg]:
               x2=indices[neg]

        if x1 < x2:
            break

    return x1, x2

image_path = 'r15.jpg'
image, gray, inverted_thresh = preprocess_image(image_path)

largest_contour = find_largest_contour(inverted_thresh)
x, y, w, h = get_bounding_box(largest_contour)
v1=y+h
v2=x+w
cropped_image = image[y:v1, 0 :]
cropped_image = cropped_image[0 : ,x : v2]
conss=100
edges = cv2.Canny(cropped_image, conss, 2*conss)

cropped_mean_intensity = np.mean(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY))
_, cropped_thresh = cv2.threshold(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY), cropped_mean_intensity, 255, cv2.THRESH_BINARY)
cropped_inverted_thresh = cv2.bitwise_not(cropped_thresh)
cropped_contours, _ = cv2.findContours(cropped_inverted_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_contour_area = 0
bright=255
largest_cropped_contour = None
for contour in cropped_contours:

    current_contour_area = cv2.contourArea(contour)
    if current_contour_area > max_contour_area:
        max_contour_area = current_contour_area
        largest_cropped_contour = contour
if largest_cropped_contour is not None:
    cropped_inverted_thresh = cv2.bitwise_not(cropped_thresh)
    markers = find_connected_components(cropped_inverted_thresh)
    neg=-1
    pos=0

    for i in range(1, np.max(markers) + 1):
        component = np.where(markers == i, bright, 0).astype(np.uint8)
        component_contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for component_contour in component_contours:
            if cv2.contourArea(component_contour) < 101:
                cv2.drawContours(cropped_inverted_thresh, [component_contour], neg, pos, neg)

x1, x2 = calculate_x_coordinates(cropped_inverted_thresh)
cropped_midpoint = (x1 + x2)
cropped_midpoint = cropped_midpoint // 2
boundary_mask = np.zeros_like(edges)
boundary_mask[y:v1, :] = bright
boundary_mask[:, x:v2] = bright
boundary_edges = cv2.bitwise_and(edges, edges, mask=boundary_mask)

cv2_imshow(boundary_edges)


cv2.line(cropped_image, (cropped_midpoint, 0), (cropped_midpoint, cropped_image.shape[0]), (bright, 0, 0), 2)
y_distance = int(0.5 * h)

cv2.line(cropped_image, (0, y_distance - 15), (cropped_image.shape[1], y_distance - 15), (bright, 0, 0), 2)
cv2.line(cropped_image, (0, y_distance + 15), (cropped_image.shape[1], y_distance + 15), (bright, 0, 0), 2)
cv2.line(cropped_image, (0, y_distance), (cropped_image.shape[1], y_distance), (bright, 0, 0), 2)
r = w - cropped_midpoint
l = cropped_midpoint

d = r / l
if 1.11999 < d < 1.149999:
        y_center = int(0.5 * h)

        fake_flag = False

        for x in range(0, w - 26 + 2):
            a = y_center - 25 // 2
            b = y_center + 15 // 2
            c=x
            d=x + 25
            window_edges = boundary_edges[a:b, c:d]
            edge_count = np.count_nonzero(window_edges == 255)
            if edge_count < 1.5:
                fake_flag = True
                break

        if fake_flag:
            print("Image is fake")
        else:
            print("Image is real")
else:
    print("fake")

cv2_imshow(cropped_image)
print("Cropped Bounding Box Coordinates: x = {}, y = {}, w = {}, h = {}".format(x, y, w, h))
print("Cropped x1: {}, Cropped x2: {}".format(x1, x2))
print("Cropped Midpoint: ", cropped_midpoint)

