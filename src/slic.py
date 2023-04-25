import cv2
import numpy as np

def slic(image, num_segments=100, compactness=10,max_iterations = 10):
    # Convert image to LAB color space
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # 将 image_lab 转换为 np.float32 类型
    image_lab = image_lab.astype(np.float32)
    # Initialize variables
    height, width = image.shape[:2]
    s=height * width / num_segments
    step = int(np.sqrt(s))
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    centers = np.zeros((len(y_coords[::step, ::step].ravel()), 5))
    centers[:, 0:2] = np.vstack((y_coords[::step, ::step].ravel(), x_coords[::step, ::step].ravel())).T
    centers[:, 2:] = image_lab[centers[:, 0].astype(int), centers[:, 1].astype(int)]
    distances = np.ones((height, width)) * np.inf
    labels = np.zeros((height, width))
    
    # Iterate until convergence
    for i in range(max_iterations):
        # Assign pixels to nearest center
        for idx, center in enumerate(centers):
            x_min = int(max(center[0] - step,0))
            x_max = int(min(center[0] + step,height))
            y_min = int(max(center[1] - step,0))
            y_max = int(min(center[1] + step,width))
            # sub_centers = centers[idx, 0:2] - np.array([y_min, x_min])
            sub_image_lab = image_lab[x_min:x_max,y_min:y_max,:]

            sub_lab_distances = np.sum((sub_image_lab - centers[idx, 2:]) ** 2, axis=2) 
            x, y = np.meshgrid(np.arange(x_min,x_max), np.arange(y_min,y_max))
            sub_idx=np.vstack((y.ravel(), x.ravel())).T
            sub_dis=np.sum((sub_idx.reshape(x_max-x_min,y_max-y_min,2)- centers[idx, 0:2])**2,axis=2)/s
            sub_distances=np.sqrt(sub_dis*compactness+sub_lab_distances)
            # sub_distances = np.sqrt(np.sum((sub_image_lab - centers[idx, 2:]) ** 2, axis=2) + np.sum(sub_centers ** 2, axis=1)[:, np.newaxis])
            labels[x_min:x_max, y_min:y_max][sub_distances < distances[x_min:x_max, y_min:y_max]] = idx
            distances[x_min:x_max, y_min:y_max][sub_distances < distances[x_min:x_max, y_min:y_max]] = sub_distances[sub_distances < distances[x_min:x_max, y_min:y_max]]
            
        
        # Update center positions
        for idx, center in enumerate(centers):
            mask = (labels == idx)
            if np.sum(mask) == 0:
                continue
            new_center = np.hstack((np.mean(x_coords[mask]), np.mean(y_coords[mask])))
            new_center_lab = np.mean(image_lab[mask], axis=0)
            centers[idx] = np.hstack((new_center, new_center_lab))
    
    # Return labels as uint8 array
    return labels.astype(np.uint8)
# def calculate_superpixel_edges(labels):
#     # Initialize edges as zeros array
#     edges = np.zeros_like(labels, dtype=np.uint8)
#     # Iterate over each pixel
#     for x in range(1, labels.shape[0]-1):
#         for y in range(1, labels.shape[1]-1):
#             # Check if current pixel is on the border of a superpixel
#             if labels[x, y] != labels[x-1, y] or labels[x, y] != labels[x+1, y] or labels[x, y] != labels[x, y-1] or labels[x, y] != labels[x, y+1]:
#                 edges[x, y] = 255
#     # Return edges as uint8 array
#     return edges
def calculate_superpixel_edges(labels):
    # Initialize edges as zeros array
    edges = np.zeros_like(labels, dtype=np.uint8)
    # Check if each pixel is on the border of a superpixel
    edges[1:-1, 1:-1] = np.logical_or.reduce((
        labels[1:-1, 1:-1] != labels[:-2, 1:-1],
        labels[1:-1, 1:-1] != labels[2:, 1:-1],
        labels[1:-1, 1:-1] != labels[1:-1, :-2],
        labels[1:-1, 1:-1] != labels[1:-1, 2:]
    ))
    # Return edges as uint8 array
    return edges * 255
# # Load input image
image = cv2.imread("white.png")
image = cv2.resize(image, (400, 400))
# # Apply SLIC algorithm to generate superpixels
superpixels = slic(image,100,compactness=10)


# Visualize superpixels
segmented_image = np.zeros(image.shape, dtype=np.uint8)
for i in range(np.max(superpixels)):
    mask = (superpixels == i)
    color = np.random.randint(0, 255, size=3)
    segmented_image[mask] = color
# 将超像素分割结果转换为灰度图像
gray = cv2.cvtColor(superpixels, cv2.COLOR_GRAY2BGR)

#求边缘
mask_slic=calculate_superpixel_edges(superpixels)
mask_inv_slic = cv2.bitwise_not(mask_slic)
# img_slic = cv2.bitwise_and(image,image,mask =  mask_slic) #在原图上绘制超像素边界
img_slic = cv2.bitwise_and(image,image,mask =  mask_inv_slic) #在原图上绘制超像素边界

cv2.imshow("Segmented Image", segmented_image)
cv2.imshow("img_slic",img_slic)
cv2.imshow("Gray Image", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
