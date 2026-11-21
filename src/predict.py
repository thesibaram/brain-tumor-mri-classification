import os
import random
from PIL import Image
import torch
import matplotlib.pyplot as plt

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_samples(test_data, test_transform, device, model, class_names=CLASS_NAMES):
    plt.figure(figsize=(16, 4))

    for i, class_name in enumerate(os.listdir(test_data)):
        class_folder = os.path.join(test_data, class_name)
        random_img = random.choice(os.listdir(class_folder))
        img_path = os.path.join(class_folder, random_img)

        img = Image.open(img_path).convert("L")
        tensor = test_transform(img).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(tensor)
            predicted = torch.argmax(outputs, dim=1)
            predicted_label = class_names[predicted.item()]
            color = "green" if predicted_label == class_name else "red"

        plt.subplot(1, 4, i+1)
        plt.imshow(img, cmap="gray")
        plt.title(f"True: {class_name}\nPred: {predicted_label}", color=color)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("../outputs/predictions/predictions.png")
    plt.show()


def predict_single(img_path, test_transform, device, model, class_names=CLASS_NAMES):
    img = Image.open(img_path).convert("L")
    tensor = test_transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        predicted = torch.argmax(outputs, dim=1)
        predicted_label = class_names[predicted.item()]

    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray")
    plt.title(f"Pred: {predicted_label}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return predicted_label