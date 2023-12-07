import cv2
import os
from pathlib import Path
from torch.utils.data import Dataset


class CropsDataset(Dataset):
    def __init__(self, suffix:str="_mask"):
        base_folder=Path(os.path.join(os.getcwd(), 'crops'))
        # Get all png images in the folder
        self.data = {path.split('/')[-1].split('.')[0]: path for path in list(map(str, base_folder.glob("*[!_mask].png")))}
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index):
        crop_path = self.data[index]
        crop = cv2.imread(crop_path)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop