import cv2
import numpy as np
from tkinter import filedialog, Tk
from sklearn.cluster import KMeans

def find_dominant_colors(image, k=4):
    reshaped_image = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(reshaped_image)
    return kmeans.cluster_centers_

def create_color_mask(image, color, threshold=30):
    lower_bound = np.maximum(color - threshold, 0)
    upper_bound = np.minimum(color + threshold, 255)
    mask = cv2.inRange(image, lower_bound, upper_bound)
    return mask

def apply_mask(image, mask):
    rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba_image[:, :, 3] = mask
    return rgba_image

def main():
    root = Tk()
    root.withdraw()

    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not image_path:
        print("No image selected.")
        return

    image = cv2.imread(image_path)
    dominant_colors = find_dominant_colors(image)

    folder_path = filedialog.askdirectory(title="Select a Folder to Save the Output Images")
    if not folder_path:
        print("No folder selected.")
        return

    for i, color in enumerate(dominant_colors):
        mask = create_color_mask(image, color)
        result = apply_mask(image, mask)
        output_path = f"{folder_path}/transparent_color_{i}.png"
        cv2.imwrite(output_path, result)
        print(f"Color isolated image with transparency saved as {output_path}")

if __name__ == "__main__":
    main()