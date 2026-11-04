import os
from datasets import load_dataset

# 1. Load the dataset from Hugging Face
print("Loading dataset...")
dataset = load_dataset("huggan/selfie2anime")

# 2. Define your target root folder
base_dir = "./cyclegan_selfie2anime"
splits = ["train", "test"]
domains = ["A", "B"]

# Create nested directories: train/A, train/B, test/A, test/B
for split in splits:
    for domain in domains:
        os.makedirs(os.path.join(base_dir, split, domain), exist_ok=True)

# 3. Process and save images into the new structure
for split in splits:
    print(f"Processing {split} split ({len(dataset[split])} samples)...")
    
    for idx, item in enumerate(dataset[split]):
        # Extract images (PIL Image objects)
        img_a = item["imageA"]  # Human selfie
        img_b = item["imageB"]  # Anime face
        
        # Define paths with train/test as the primary folder
        path_a = os.path.join(base_dir, split, "A", f"{idx:05d}.jpg")
        path_b = os.path.join(base_dir, split, "B", f"{idx:05d}.jpg")
        
        # Save as JPEG (convert to RGB in case of transparency)
        img_a.convert("RGB").save(path_a, "JPEG")
        img_b.convert("RGB").save(path_b, "JPEG")

print(f"Successfully saved all images to '{base_dir}/' with train/test folders!")