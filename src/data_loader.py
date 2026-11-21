from torch.utils.data import Dataset, random_split, Subset, DataLoader
import os
from PIL import Image
import torch

class BTdataset(Dataset):
    def __init__(self, root, transform= None):
        self.root= root
        self.transform= transform
        self.classes = sorted(os.listdir(root))
        self.samples= []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root, class_name)

            if not os.path.isdir(class_dir):
                continue
            for img in os.listdir(class_dir):
                img_path= os.path.join(class_dir, img)
                img= Image.open(img_path).convert("L").resize((224, 224))
                self.samples.append((img, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(
        0,
        translate= (0.05, 0.05),
        scale= (0.95, 1.05)
    ),
    transforms.ColorJitter(
        brightness= 0.1,
        contrast= 0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean= [0.183], std= [0.200])
])

val_transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean= [0.183], std= [0.200])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean= [0.183], std= [0.200])
])

def get_dataloaders(train_data, test_data, train_transform, val_transform, test_transform):
    train_dataset= BTdataset(train_data, train_transform)
    val_dataset= BTdataset(train_data, val_transform)
    test_dataset= BTdataset(test_data, test_transform)

    total_size= len(train_dataset)
    train_size= int(len(train_dataset)) * 0.9
    val_size= total_size - train_size

    generator= torch.Generator().manual_seed(42)

    train_split, val_split= random_split(
        train_dataset,
        [train_size, val_size],
        generator= generator
    )

    train_subset= Subset(train_dataset, train_split.indices)
    val_subset= Subset(val_dataset, val_split.indices)

    train_loader= DataLoader(
        dataset= train_subset,
        batch_size= 64,
        shuffle= True,
        num_workers= 0,
        pin_memory= True
    )

    val_loader= DataLoader(
        dataset= val_subset,
        batch_size= 64,
        shuffle= False,
        num_workers= 0,
        pin_memory= True
    )
    test_loader= DataLoader(
        dataset= test_dataset,
        batch_size= 64,
        shuffle= False,
        num_workers= 0,
        pin_memory= True
    )

    return train_loader, val_loader, test_loader





