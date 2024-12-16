import cv2
import numpy as np
import matplotlib.pyplot as plt


# Define the image path dynamically
import os
image_name = "motherboard_image.jpeg"
base_dir = "C:/Users/hassa/Downloads/850 Project 3"
image_path = os.path.join(base_dir, image_name)

# Load and preprocess the image
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Rotate for correct orientation
    image_rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Convert to grayscale
    gray = cv2.cvtColor(image_rotated, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    return image_rotated, blurred

# Apply thresholding and detect edges
def threshold_and_detect_edges(blurred):
    _, thresholded = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresholded, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    return edges_dilated

# Extract the largest contour and mask the image
def extract_largest_contour(image, blurred, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise Exception("No contours found - check edge detection parameters.")

    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(blurred, dtype="uint8")  # Use blurred as reference size
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    # Apply the mask to the original image
    extracted = cv2.bitwise_and(image, image, mask=mask)

    return mask, extracted

# Visualize results
def visualize_results(image, edges, extracted):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Edge Detection")
    plt.imshow(edges, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Extracted Motherboard")
    plt.imshow(cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Main workflow
output_name = "extracted_pcb_clean.JPEG"
output_path = os.path.join(base_dir, output_name)

image, blurred = load_and_preprocess_image(image_path)
edges = threshold_and_detect_edges(blurred)
mask, extracted_pcb = extract_largest_contour(image, blurred, edges)

# Save the extracted PCB
cv2.imwrite(output_path, extracted_pcb)

# Visualize results
visualize_results(image, edges, extracted_pcb)
