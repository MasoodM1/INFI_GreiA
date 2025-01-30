import os
from PIL import Image
import numpy as np

class ImageResizer:
    def __init__(self, input_folder, output_folder, size=(180, 180)):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.size = size
        os.makedirs(output_folder, exist_ok=True)

    def resize_images(self):
        for filename in os.listdir(self.input_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.input_folder, filename)
                img = np.array(Image.open(img_path).resize(self.size))
                output_path = os.path.join(self.output_folder, filename)
                Image.fromarray(img).save(output_path)
                print(f"Resized and saved {filename} to {self.output_folder}")
