from PIL import Image
import os
import numpy as np

folders = ['Schere', 'Stein', 'Papier']
output_folder = 'Aufgabe_06_CNN_SchereSteinPapier/A06_CNN/img_resize/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for folder in folders:
    folder_path = f'Aufgabe_06_CNN_SchereSteinPapier/A06_CNN/img/{folder}/'
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = np.array(Image.open(img_path).resize((180, 180)))
            output_path = os.path.join(output_folder, f'{folder}_{filename}')
            Image.fromarray(img).save(output_path)