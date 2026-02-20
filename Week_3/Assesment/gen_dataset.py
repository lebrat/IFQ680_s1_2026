import os
from tqdm import tqdm
from torchvision.datasets import FGVCAircraft
import torchvision.transforms as transforms
import numpy as np

def remove_banner(img):
    width, height = img.size
    # Crop: (left, top, right, bottom)
    # This keeps the image from the top (0) to 20 pixels before the bottom
    return img.crop((0, 0, width, height - 20))

selected_classes = [11, 12, 18, 20, 23, 33, 38, 39, 44, 45, 53, 60, 64, 67, 68, 70, 74, 75, 76, 86]

# 2. Setup paths
output_root = "FGVCAircraft_Subset20"
splits = ["trainval", "test"]

for split in splits:
    print(f"Processing {split} split...")
    dataset = FGVCAircraft(root="/Users/fonseca2@qut.edu.au/Projects/IFQ680_s1_2026/Week_3/Assesment/data", split=split, download=True, transform=transforms.Lambda(remove_banner))
    all_labels = np.array(dataset._labels)
    
    for c, class_idx in enumerate(selected_classes):
        for i, sample_idx in enumerate(np.where(all_labels == class_idx)[0]):
    
            # Access raw image and label index           
            img, label = dataset[sample_idx]            
            assert label == class_idx

            class_name = f"class_{c:02d}"
            target_dir = os.path.join(output_root, split, class_name)            
            os.makedirs(target_dir, exist_ok=True)
            img.save(os.path.join(target_dir, f"img_{i:04d}.jpg"))
