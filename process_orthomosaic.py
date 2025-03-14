import rasterio
from rasterio.windows import Window
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

###############################################################
# Parameters
###############################################################
filename = "./orthomosaics/Cropped_field.tif"
window_height = 500
window_width = 500
std_deviations = 2



###############################################################
# Get mean and standard deviation of annoted pixels
###############################################################
img = cv2.imread("./resources/exercise_1/EB-02-660_0595_0186.JPG")
annotated_img = cv2.imread("./resources/exercise_1/EB-02-660_0595_0186_annoted.JPG")
img_masked_BGR = cv2.inRange(annotated_img, (0, 0, 240), (0, 0, 255))
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
annot_pix_values_lab = img_lab[img_masked_BGR == 255]
mean_LAB = np.mean(annot_pix_values_lab, axis=0)
std_dev_LAB = np.std(annot_pix_values_lab, axis=0)
cov = np.cov(annot_pix_values_lab.T)
print("\Anoted pixels")
print("Mean CieLAB: ", mean_LAB)
print("StD CieLAB: ", std_dev_LAB)



###############################################################
# Orthomosaic windowing
###############################################################
src = rasterio.open(filename)
# Get orthomosaic dimensions
ortho_height = src.height
ortho_width = src.width 
print("\nOrthomosaic:") 
print("Dimensions: ", ortho_height, ortho_width)

current_x = 0
current_y = 0

pumpkin_count = 0
on_edge = 0

while True:
    # Read in orthomosaic
    img = src.read(window=Window(current_x, current_y, window_height, window_width))
    print(current_x, current_y)
    # Convert to opencv image format
    temp = img.transpose(1, 2, 0)
    t2 = cv2.split(temp)
    img_cv = cv2.merge([t2[2], t2[1], t2[0]])

    # Count pumpkins
    # Convert to CieLAB
    segmented_image_lab = np.zeros(img_cv.shape, dtype=np.uint8)
    img_lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    # Apply gaussian blur to remove noise in the image
    img_lab = cv2.GaussianBlur(img_lab, (5, 5), 0)
    # Calculate mahalanobis distance
    pixels = np.reshape(img_lab, (-1, 3))   
    inv_cov = np.linalg.inv(cov)
    diff = pixels - mean_LAB
    moddotproduct = diff * (diff @ inv_cov)
    mahalanobis_dist = np.sum(moddotproduct,axis=1)
    distance = np.sqrt(np.reshape(mahalanobis_dist,(img_lab.shape[0],img_lab.shape[1])))
    # Normalize the distance
    #normalizedImg = np.zeros((img_lab.shape[0], img_lab.shape[1]))
    #normalizedImg = cv2.normalize(distance, normalizedImg, 0, 255, cv2.NORM_MINMAX)
    # Histogram of distances
    #plt.figure()
    #sns.histplot(distance.flatten())
    # Save histogram
    #plt.savefig("./output/exercise_3/histogram.jpg")
    #plt.close()
    # Threshold the image based on a distance threshold
    threshold = 3.1
    thresholded_img = np.where(distance < threshold, 255, 0)
    #thresholded_img = np.where(normalizedImg < threshold, 255, 0)
    thresholded_img = thresholded_img.astype(np.uint8)
    # Apply gaussian blur again
    thresholded_img = cv2.GaussianBlur(thresholded_img, (3, 3), 0)

    #cv2.imwrite("./output/exercise_3/blurred_image.jpg", img_lab)
    #cv2.imwrite("./output/exercise_3/mahalanobis_distance_image.jpg", distance)
    #cv2.imwrite("./output/exercise_3/normalized_image.jpg", thresholded_img)

    # Count white contours
    contours, _ = cv2.findContours(thresholded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = img_cv.copy()

    # Mark contours where there are multiple pumpkins in the area and count pumpkins
    for i, contour in enumerate(contours):
        # Check size of contour
        if cv2.contourArea(contour) < 10:
            continue

        # Check if contour is on edge of image
        if np.any(contour == 0) or np.any(contour == window_width) or np.any(contour == window_height):
            on_edge += 1

        if cv2.contourArea(contour) > 450:
            cv2.drawContours(img_contours, [contour], -1, (0, 0, 255), 3)
            pumpkin_count += 3
        elif cv2.contourArea(contour) > 240:
            cv2.drawContours(img_contours, [contour], -1, (255, 0, 0), 3)
            pumpkin_count += 2
        else:
            cv2.drawContours(img_contours, [contour], -1, (0, 255, 0), 3)
            pumpkin_count += 1

    print("\nCumulative pumpkin count: ", pumpkin_count)

    # Opencv show imgag
    #cv2.imshow("Orthomosaic", thresholded_img)
    #cv2.waitKey(0)
    # Move to next window
    current_x += window_width
    if current_x >= ortho_width:
        current_x = 0
        current_y += window_height
    if current_y >= ortho_height:
        break

print("\nNumber of contours on edge: ", on_edge)

pumpkin_count = pumpkin_count - int(on_edge/2)
print("\nTotal pumpkin count: ", pumpkin_count)
