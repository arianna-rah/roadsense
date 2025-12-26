import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

label_map = {
    **{c: "dry" for c in ["dry_asphalt_severe", "dry_asphalt_slight", "dry_asphalt_smooth", "dry_concrete_severe", "dry_concrete_slight", "dry_concrete_smooth", "dry_gravel", "dry_mud"]},
    **{c: "wet" for c in ["wet_asphalt_severe", "wet_asphalt_slight", "wet_asphalt_smooth", "wet_concrete_severe", "wet_concrete_slight", "wet_concrete_smooth", "wet_gravel", "wet_mud"]},
    **{c: "standing_water" for c in ["water_asphalt_severe", "water_asphalt_slight", "water_asphalt_smooth", "water_concrete_severe", "water_concrete_slight", "water_concrete_smooth", "water_gravel", "water_mud"]},
    **{c: "snow" for c in ["fresh_snow", "melted_snow"]},
    **{c: "ice" for c in ["ice"]}
}

class_to_idx = {"dry": 0, "wet": 1, "standing_water": 2, "snow": 3, "ice": 4}
class_names = ["dry", "wet", "standing_water", "snow", "ice"]

dataset_path_train = "/kaggle/input/rscd-dataset-1million/RSCD dataset-1million/train"
dataset_path_test = "/kaggle/input/rscd-dataset-1million/RSCD dataset-1million/train"



class RSCDDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, label_map=None, class_names=None, class_to_idx=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.label_map = label_map
        self.class_names = class_names
        self.class_to_idx = class_to_idx
        self.samples = []
    
        for original_label in self.root_dir.iterdir():
            if original_label.isdir() and original_label.name in self.label_map:
                mapped_label = self.label_map[original_label.name]
                label_idx  = self.class_to_idx[mapped_label]

            for img_path in original_label.glob('*.jpg'):
                self.samples.append((str(img_path), label_idx, mapped_label))
    
    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        img_path, label_idx, label_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB') # don't know why .convert() is used. look into?
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx
            
def find_distribution(dataset):
    labels = [sample[1] for sample in dataset.samples]
    unique, counts = np.unique(labels, return_counts=True) # finds duplicates and removes them, returns unique labels and number of images in each label class
    print("Class distribution: ")
    for idx, count in zip(unique, counts):
        print(f"{idx}/{dataset.class_names[idx]}: {count} images ({count/len(labels)*100:.2f}%)")

    return dict(zip(unique, counts))

train_transform = 

test_transform =
