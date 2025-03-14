import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

############################################
# Exercise 1
############################################
# Load in original image
img = cv2.imread("./resources/exercise_1/EB-02-660_0595_0186.JPG")
print("Original Image Shape: ", img.shape)

# Load in annoted image
annotated_img = cv2.imread("./resources/exercise_1/EB-02-660_0595_0186_annoted.JPG")
print("Annotated Image Shape: ", annotated_img.shape)

# Mask red pixels in the original image
img_masked_BGR = cv2.inRange(annotated_img, (0, 0, 240), (0, 0, 255))

# Show pixel coordinates for white pixels
white_pixels = np.where(img_masked_BGR == 255)
print("Red pixel found: ", len(white_pixels[0]))

# Save the image
print("\nCreating mask from red pixels")
cv2.imwrite("./output/exercise_1/mask.jpg", img_masked_BGR)
print("Mask saved to ./output/exercise_1/mask.jpg")

# Get the mean BGR pixel values of the pixels in the mask
annot_pix_values = img[img_masked_BGR == 255]
mean_BGR = np.mean(annot_pix_values, axis=0)
std_dev_BGR = np.std(annot_pix_values, axis=0)
print("\nBGR color space")
print("Mean BGR: ", mean_BGR)
print("StD BGR: ", std_dev_BGR)

# Convert to CieLAB
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# show img_lab
annot_pix_values_lab = img_lab[img_masked_BGR == 255]
mean_LAB = np.mean(annot_pix_values_lab, axis=0)
std_dev_LAB = np.std(annot_pix_values_lab, axis=0)
print("\nCieLAB color space")
print("Mean LAB: ", mean_LAB)
print("StD LAB: ", std_dev_LAB)

# Visualise the distribution of colour values
# BGR
plt.figure()
sns.histplot(annot_pix_values[:, 0], color='b', label='B')
sns.histplot(annot_pix_values[:, 1], color='g', label='G')
sns.histplot(annot_pix_values[:, 2], color='r', label='R')
plt.title("BGR Color Space")
plt.legend()
plt.xlabel("Pixel Value")
plt.savefig("./output/exercise_1/BGR_histogram.png")

# CieLAB
plt.figure()
sns.histplot(annot_pix_values_lab[:, 0], color='b', label='L')
sns.histplot(annot_pix_values_lab[:, 1], color='g', label='A')
sns.histplot(annot_pix_values_lab[:, 2], color='r', label='B')
plt.title("CieLAB Color Space")
plt.legend()
plt.xlabel("Pixel Value")
plt.savefig("./output/exercise_1/LAB_histogram.png")

############################################
# Exercise 2
############################################
# Segment the image that are within a certain distance of the mean color of the masked pixels with cv2.inRange()
print("\nSegmenting the images based on colors in masked pixels with cv2.inRange()")
segmented_image = np.zeros(img.shape, dtype=np.uint8)
segmented_image_lab = np.zeros(img.shape, dtype=np.uint8)

std_deviations = 2
segmented_image = cv2.inRange(img, mean_BGR - std_deviations*std_dev_BGR, mean_BGR + std_deviations*std_dev_BGR)
segmented_image_lab = cv2.inRange(img_lab, mean_LAB - std_deviations*std_dev_LAB, mean_LAB + std_deviations*std_dev_LAB)

# Save the images
cv2.imwrite("./output/exercise_2/segmented_image_bgr.jpg", segmented_image)
cv2.imwrite("./output/exercise_2/segmented_image_lab.jpg", segmented_image_lab)
print("Segmented images saved to ./output/exercise_2/")

# Segment the image based on mahalanobis distance
print("\nCalculating covariance matrix")
cov = np.cov(annot_pix_values_lab.T)
print("Covariance matrix:\n", cov)

Cref = mean_LAB

# Apply gaussian blur to the image
print("\nApplying Gaussian blur to the image")
img_lab = cv2.GaussianBlur(img_lab, (5, 5), 0)
# Save image
cv2.imwrite("./output/exercise_2/blurred_image.jpg", img)
print("Blurred image saved to ./output/exercise_2/")

# Calculate mahalanobis distance
'''distance = np.zeros(img.shape[:2])
print("\nCalculating Mahalanobis distance")
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        C = img[i, j]
        distance[i, j] = (C - Cref).T @ cov_inv @ (C - Cref)
    print("Iteration: ", i, " / ", img.shape[0], end="\r")
distance = np.sqrt(distance)'''

print("\nCalculating Mahalanobis distance")
pixels = np.reshape(img_lab, (-1, 3))   
inv_cov = np.linalg.inv(cov)
diff = pixels - mean_LAB
moddotproduct = diff * (diff @ inv_cov)
mahalanobis_dist = np.sum(moddotproduct,
axis=1)
distance = np.sqrt(np.reshape(mahalanobis_dist,(img_lab.shape[0],img_lab.shape[1])))


# Normalize the distance
normalizedImg = np.zeros((img_lab.shape[0], img_lab.shape[1]))
normalizedImg = cv2.normalize(distance, normalizedImg, 0, 255, cv2.NORM_MINMAX)

# Save image
cv2.imwrite("./output/exercise_2/mahalanobis_distance_image.jpg", normalizedImg)
print("Mahalanobis distance image saved to ./output/exercise_2/")

# Plot histogram of the distance image
'''plt.figure()
sns.histplot(normalizedImg.ravel(), color='b')
plt.title("Mahalanobis Distance Histogram")
plt.xlabel("Distance")
plt.savefig("./output/exercise_2/mahalanobis_distance_histogram.png")
print("Mahalanobis distance histogram saved to ./output/exercise_2/")'''


# Threshold the image based on a distance threshold
print("\nThresholding the image based on a distance threshold")
threshold = 63
thresholded_img = np.where(normalizedImg < threshold, 255, 0)
cv2.imwrite("./output/exercise_2/segmented_image_mahalanobis.jpg", thresholded_img)
print("Segmented image based on Mahalanobis distance saved to ./output/exercise_2/")

# Convert to CV_8UC1
thresholded_img = thresholded_img.astype(np.uint8)

# Apply gaussian blur again
print("\nApplying Gaussian blur to the thresholded image")
thresholded_img = cv2.GaussianBlur(thresholded_img, (3, 3), 0)
cv2.imwrite("./output/exercise_2/segmented_image_mahalanobis_blurred.jpg", thresholded_img)

# Count white countours using cv2
print("\nCounting white contours")
contours, _ = cv2.findContours(thresholded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("\nNumber of white contours: ", len(contours))

# Draw contours on the original image
img_contours = img.copy()

# Mark contours where there are multiple pumpkins in the area
print("\nMarking contours where there are multiple pumpkins in the area")

pumpkin_count = 0
for i, contour in enumerate(contours):
    if cv2.contourArea(contour) > 450:
        cv2.drawContours(img_contours, [contour], -1, (0, 0, 255), 3)
        pumpkin_count += 3
    elif cv2.contourArea(contour) > 240:
        cv2.drawContours(img_contours, [contour], -1, (255, 0, 0), 3)
        pumpkin_count += 2
    else:
        cv2.drawContours(img_contours, [contour], -1, (0, 255, 0), 3)
        pumpkin_count += 1

print("\n Total number of pumpkins: ", pumpkin_count)
cv2.imwrite("./output/exercise_2/contours.jpg", img_contours)

# Mark pumpkins on the original image with a circle
print("\nMarking pumpkins on the original image")
for i, contour in enumerate(contours):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(img, center, radius, (0, 0, 255), 3)

cv2.imwrite("./output/exercise_2/pumpkins_marked.jpg", img)
print("Pumpkins marked on the original image saved to ./output/exercise_2/")
