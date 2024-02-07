import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from sklearn.cluster import KMeans

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class ColorExtractorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Color Extractor App")
        self.geometry("500x300")

        ctk.set_appearance_mode("Dark") # You can change this to light if you're a freak <3
        ctk.set_default_color_theme("blue")

        self.create_widgets()

    def create_widgets(self):
        self.select_image_button = ctk.CTkButton(self, text="Select Image", command=self.select_image)
        self.select_image_button.pack(pady=20)

        self.colors_label = ctk.CTkLabel(self, text="Number of Dominant Colors:")
        self.colors_label.pack()
        self.num_colors = ctk.CTkEntry(self)
        self.num_colors.pack(pady=10)

        self.bg_color_label = ctk.CTkLabel(self, text="Background Color:")
        self.bg_color_label.pack()
        self.bg_color = ctk.CTkComboBox(self, values=["White", "Black"])
        self.bg_color.set("White")
        self.bg_color.pack(pady=10)

        self.start_button = ctk.CTkButton(self, text="Process Image", command=self.process_image)
        self.start_button.pack(pady=20)

    def select_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            print(f"Selected image: {self.image_path}")

    def process_image(self):
        if not self.image_path:
            print("No image selected.")
            return

        try:
            num_colors = int(self.num_colors.get())
        except ValueError:
            print("Invalid number of colors.")
            return

        fill_color = WHITE if self.bg_color.get() == "White" else BLACK
        image = cv2.imread(self.image_path)
        dominant_colors = self.find_dominant_colors(image, num_colors)

        folder_path = filedialog.askdirectory()
        if not folder_path:
            print("No folder selected.")
            return

        for i, color in enumerate(dominant_colors):
            mask = self.create_color_mask(image, color)
            result = self.apply_mask(image, mask, fill_color)
            output_path = f"{folder_path}/color_group_{i}.png"
            cv2.imwrite(output_path, result)
            print(f"Saved: {output_path}")

    def create_color_mask(self, image, color, threshold=30):
        lower_bound = np.maximum(color - threshold, 0)
        upper_bound = np.minimum(color + threshold, 255)
        mask = cv2.inRange(image, lower_bound, upper_bound)
        blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        smooth_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_CLOSE, kernel)
        return smooth_mask

    def apply_mask(self, image, mask, fill_color):
        filled_image = np.zeros_like(image)
        filled_image[:] = fill_color
        filled_image[mask != 0] = image[mask != 0]
        return filled_image

    def find_dominant_colors(self, image, k):
        reshaped_image = image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(reshaped_image)
        return kmeans.cluster_centers_

if __name__ == "__main__":
    app = ColorExtractorApp()
    app.mainloop()
