import torch
from torch.utils.data import Dataset

from PIL import Image

class ChemicalDataset(Dataset):
    def __init__(self, dataframe, image_path, transform=None, target = True):
        super().__init__()
        
        self.image_paths = dataframe[image_path]
        self.transform   = transform
        self.targets     = target
        
        if self.targets:
            self.targets = dataframe['InChI_tokenized']
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        
        image = Image.open(self.image_paths[idx])

        if self.transform:
            image = self.transform(image)
        
        data = {'images': image}
        
        if self.targets is not False:
            data['tokens'] = torch.LongTensor(self.targets[idx].ids)
            data['attention_mask'] = torch.LongTensor(self.targets[idx].attention_mask)
        
        return data