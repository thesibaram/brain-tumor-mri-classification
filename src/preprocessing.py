import os
from PIL import Image
import numpy as np

def compute_mean_std(train_data):
    all_images = []
    for class_name in os.listdir(train_data):
        class_folder= os.path.join(train_data, class_name)
        for image in os.listdir(class_folder):
            img_path= os.path.join(class_folder, image)

            img= Image.open(img_path).convert("L").resize((224, 224))
            img= np.array(img)/255.0
            all_images.append(img)

    all_images= np.array(all_images)
    mean = all_images.mean(axis= (0, 1, 2 ))
    std = all_images.std(axis= (0, 1, 2 ))
    print(f"Mean: {mean:.4f} | Std: {std:.4f}")
    return mean, std

