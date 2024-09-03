import torch
import timm
import cv2
from tqdm import tqdm
import numpy as np

# Load the MiDaS model
model_type = "DPT_Large"  # Use DPT_Large, DPT_Hybrid, or MiDaS_small for smaller models
model = torch.hub.load("intel-isl/MiDaS", model_type)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load the transformation functions from MiDaS
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


# Load and preprocess the image
img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply MiDaS transformation
input_batch = transform(img).to(device)

# Run the model
with torch.no_grad():
    prediction = model(input_batch)

# Resize the output to match the input image size
prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=img.shape[:2],
    mode="bicubic",
    align_corners=False,
).squeeze()

# Convert to numpy array and normalize the depth map for visualization
depth_map = prediction.cpu().numpy()
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

import matplotlib.pyplot as plt

# Display the depth map
plt.imshow(depth_map, cmap='plasma')
plt.colorbar()
plt.show()

import numpy as np
import cv2


def generate_stereo_images(img, depth_map, max_disparity=30):
    """
    Generates left and right images for stereo vision from a depth map.

    :param img: The original image (RGB).
    :param depth_map: The depth map (normalized to 0-1).
    :param max_disparity: Maximum disparity (shift) value for the stereo effect.
    :return: Left and right images.
    """
    # Normalize depth map to range between 0 and 1 if not already done
    depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    print("depth_map normalized")
    # Convert depth map to disparity (shift) map
    disparity_map = (1 - depth_map) * max_disparity
    print("disparity map created")
    # Initialize the left and right images
    h, w, _ = img.shape
    left_image = np.zeros_like(img)
    right_image = np.zeros_like(img)
    print("images initialized")

    # Add a progress bar to the loop
    for y in tqdm(range(h), desc="Generating Stereo Images", unit="row"):
        for x in range(w):
            shift = int(disparity_map[y, x])
            if x - shift >= 0:
                left_image[y, x] = img[y, x - shift]
            else:
                left_image[y, x] = img[y, 0]

            if x + shift < w:
                right_image[y, x] = img[y, x + shift]
            else:
                right_image[y, x] = img[y, w - 1]

    return left_image, right_image


def generate_stereo_images_with_inpainting(img, depth_map, max_disparity=30):
    # Normalize depth map to range between 0 and 1 if not already done
    depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    print("depth_map normalized")
    # Convert depth map to disparity (shift) map
    disparity_map = (1 - depth_map) * max_disparity
    print("disparity map created")

    h, w, _ = img.shape
    left_image = np.zeros_like(img)
    right_image = np.zeros_like(img)
    left_mask = np.zeros((h, w), dtype=np.uint8)
    right_mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            shift = int(disparity_map[y, x])
            if x - shift >= 0:
                left_image[y, x] = img[y, x - shift]
            else:
                left_mask[y, x] = 255

            if x + shift < w:
                right_image[y, x] = img[y, x + shift]
            else:
                right_mask[y, x] = 255

    # Inpaint the missing areas
    left_image = cv2.inpaint(left_image, left_mask, 3, cv2.INPAINT_TELEA)
    right_image = cv2.inpaint(right_image, right_mask, 3, cv2.INPAINT_TELEA)

    return left_image, right_image


# Generate the stereo images
left_image, right_image = generate_stereo_images_with_inpainting(img, depth_map, max_disparity=30)

# Save or display the stereo images
cv2.imwrite("left_image.jpg", cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR))
cv2.imwrite("right_image.jpg", cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR))

# Optionally display the images using OpenCV
cv2.imshow('Left Image', cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR))
cv2.imshow('Right Image', cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()